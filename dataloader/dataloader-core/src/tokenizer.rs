use crate::trie::IndexedTrie;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::{self, BufRead};

extern crate regex;
use regex::Regex;

pub type Token = i64;

pub struct OpenIndexer<T>
where
    T: Eq + Hash + Clone,
{
    next_idx: i64,
    map: HashMap<T, i64>,
    frozen: bool,
}

pub type PickleableIndexer<T> = (i64, HashMap<T, i64>, bool);

impl<T> OpenIndexer<T>
where
    T: Eq + Hash + Clone + Display,
{
    pub fn new() -> Self {
        OpenIndexer {
            next_idx: 1,
            map: HashMap::new(),
            frozen: false,
        }
    }
    pub fn freeze(&mut self) {
        self.frozen = true;
    }
    pub fn add(&mut self, v: T) {
        if !self.map.contains_key(&v) {
            self.map.insert(v.clone(), self.next_idx);
            self.next_idx += 1;
        }
    }
    pub fn lookup(&self, v: T) -> i64 {
        self.map.get(&v).unwrap().copied()
    }
    pub fn reverse_lookup(&self, i: i64) -> T {
        self.map
            .iter()
            .find(|(_item, idx)| **idx == i)
            .expect("That token doesn't exist!")
            .0
            .clone()
    }
    pub fn to_pickleable(self) -> PickleableIndexer<T> {
        (self.next_idx, self.map, self.frozen)
    }
    pub fn from_pickleable(tup: PickleableIndexer<T>) -> Self {
        OpenIndexer {
            next_idx: tup.0,
            map: tup.1,
            frozen: tup.2,
        }
    }
    pub fn num_indices(&self) -> i64 {
        self.next_idx
    }
    pub fn get_all_tactics(&self) -> Vec<String> {
	(1..self.next_idx).map(|i| format!("{}", self.reverse_lookup(i))).collect()
    }
    pub fn save_to_text(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        for i in 1..self.next_idx {
            file.write(format!("{}\n", self.reverse_lookup(i)).as_bytes())
                .unwrap();
        }
    }
    pub fn load_from_text(path: &str) -> OpenIndexer<String> {
        let mut indexer = OpenIndexer::<String>::new();
        let stems = io::BufReader::new(
            File::open(path).expect(&format!("Couldn't find stem file \"{}\"", path)),
        )
        .lines()
        .map(|stem| stem.unwrap());
        for stem in stems {
            indexer.map.insert(stem, indexer.next_idx);
            indexer.next_idx += 1;
        }
        indexer
    }
}

#[pyclass(module="dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct IdentChunkTokenizer {
    use_unknowns: bool,
    subword_vocab_size: i64,
    tok_trie: IndexedTrie,
    keywords: HashMap<String, i64>,
}
pub type IdentChunk = (Token, Vec<Token>);

impl IdentChunkTokenizer {
    pub fn new_from_files(use_unknowns: bool, num_reserved_tokens: usize,
                          keywords_filepath: &str, subwords_filepath: &str) -> Self {

        let keywords: Vec<_> = io::BufReader::new(File::open(keywords_filepath).expect(&format!(
            "Couldn't open keywords file \"{}\"",
            keywords_filepath
        )))
        .lines()
        .map(|keyword| keyword.unwrap())
        .collect();
        let subwords: Vec<_> = io::BufReader::new(File::open(subwords_filepath).expect(&format!(
            "Couldn't open keywords file \"{}\"",
            subwords_filepath
        )))
        .lines()
        .map(|subword| subword.unwrap())
        .collect();
        Self::new_from_vocabs(use_unknowns, num_reserved_tokens, keywords, &subwords)
    }
    pub fn new_from_vocabs(use_unknowns: bool, num_reserved_tokens: usize,
                           keywords: Vec<String>, subwords: &Vec<String>) -> Self {
        let mut trie = IndexedTrie::new();
        let mut next_vocab_idx = num_reserved_tokens as i64;
        for item in subwords {
            trie.insert(next_vocab_idx, &item);
            next_vocab_idx += 1;
        }
        let mut keywords_map = HashMap::new();
        for (idx, keyword) in keywords.into_iter().enumerate() {
            assert!(!keywords_map.contains_key(&keyword), "Duplicate keyword {}", &keyword);
            keywords_map.insert(keyword, (idx+2) as i64);
        }

        IdentChunkTokenizer {
            use_unknowns: use_unknowns,
            subword_vocab_size: next_vocab_idx + if use_unknowns { 1 } else { 0 },
            tok_trie: trie,
            keywords: keywords_map,
        }
    }
    pub fn num_subwords(&self) -> i64 {
        self.subword_vocab_size
    }
    pub fn num_keywords(&self) -> i64 {
        (self.keywords.len() + 2) as i64
    }
    pub fn tokenize(&self, sentence: &str) -> Vec<(Token, Vec<Token>)> {
        let mut tokens = Vec::new();
        let words = get_symbols(sentence);
        for word in words {
            let mut word_tokens = Vec::new();
            let mut cur_pos = 0;
            while cur_pos < word.len() {
                match self.tok_trie.longest_prefix(&word[cur_pos..]) {
                    Some((idx, prefix)) => {
                        word_tokens.push(idx);
                        cur_pos += prefix.len()
                    }
                    None => {
                        if self.use_unknowns {
                            word_tokens.push(self.subword_vocab_size - 1)
                        }
                        cur_pos += 1
                    }
                }
            }
            match self.keywords.get(word) {
                Some(kidx) => tokens.push((*kidx, word_tokens)),
                None => {
                    tokens.push((1, word_tokens));
                }
            }
        }
        tokens
    }
}

#[pyclass(module="dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct PyIdentChunkTokenizer {
    pub inner: Option<IdentChunkTokenizer>,
}

#[pymethods]
impl PyIdentChunkTokenizer {
    #[new]
    pub fn pynew() -> Self {
        PyIdentChunkTokenizer{inner: None}
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let des: PyIdentChunkTokenizer = deserialize(s.as_bytes()).unwrap();
                self.inner = des.inner;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }
}
impl PyIdentChunkTokenizer {
    pub fn new(inner: IdentChunkTokenizer) -> Self {
        PyIdentChunkTokenizer{inner: Some(inner)}
    }
}

#[pyclass(module="dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct LongestMatchTokenizer {
    use_unknowns: bool,
    space_token: Option<i64>,
    vocab_size: i64,
    tok_trie: IndexedTrie,
}

#[pymethods]
impl LongestMatchTokenizer {
    #[new]
    pub fn pynew() -> Self {
        Self::new_from_vocab(false, false, 0, vec![])
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let result: Self = deserialize(s.as_bytes()).unwrap();
                self.use_unknowns = result.use_unknowns;
                self.space_token  = result.space_token;
                self.vocab_size = result.vocab_size;
                self.tok_trie = result.tok_trie;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()).to_object(py))
    }
}
impl LongestMatchTokenizer {
    pub fn new(use_unknowns: bool, use_spaces: bool,
               num_reserved_tokens: usize, keywords_filepath: &str) -> Self {
        let vocab: Vec<_> = io::BufReader::new(File::open(keywords_filepath).expect(&format!(
            "Couldn't open keywords file \"{}\"",
            keywords_filepath
        )))
        .lines()
        .map(|keyword| keyword.unwrap())
        .collect();
        Self::new_from_vocab(use_unknowns, use_spaces, num_reserved_tokens, vocab)
    }
    pub fn new_from_vocab(
        use_unknowns: bool,
        use_spaces: bool,
        num_reserved_tokens: usize,
        vocab: Vec<String>,
    ) -> Self {
        let mut trie = IndexedTrie::new();
        let mut next_vocab_idx = num_reserved_tokens as i64 +
            if use_spaces { 1 } else { 0 };
        for item in vocab {
            trie.insert(next_vocab_idx, &item);
            next_vocab_idx += 1;
        }
        LongestMatchTokenizer {
            use_unknowns: use_unknowns,
            space_token: if use_spaces { Some(num_reserved_tokens as i64) } else { None },
            vocab_size: next_vocab_idx + if use_unknowns { 1 } else { 0 },
            tok_trie: trie,
        }
    }
    pub fn num_tokens(&self) -> i64 {
        self.vocab_size
    }
    pub fn tokenize(&self, sentence: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let words = get_symbols(sentence);
        for word in words {
            let mut word_tokens = Vec::new();
            let mut cur_pos = 0;
            while cur_pos < word.len() {
                match self.tok_trie.longest_prefix(&word[cur_pos..]) {
                    Some((idx, prefix)) => {
                        word_tokens.push(idx);
                        cur_pos += prefix.len()
                    }
                    None => {
                        if self.use_unknowns {
                            word_tokens.push(self.vocab_size - 1)
                        }
                        cur_pos += 1
                    }
                }
            }
            tokens.append(&mut word_tokens);
            match self.space_token {
                Some(tok) => tokens.push(tok),
                None => (),
            }
        }
        tokens
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct Tokenizer {
    use_unknowns: bool,
    num_reserved_tokens: usize,
    unknown_token: Token,
    token_dict: HashMap<String, Token>,
}

pub type PickleableTokenizer = (bool, usize, Token, HashMap<String, Token>);

impl Tokenizer {
    pub fn new(use_unknowns: bool, num_reserved_tokens: usize, keywords_filepath: &str) -> Self {
        let keywords: Vec<_> = io::BufReader::new(File::open(keywords_filepath).expect(&format!(
            "Couldn't open keywords file \"{}\"",
            keywords_filepath
        )))
        .lines()
        .map(|keyword| keyword.unwrap())
        .collect();
        let first_token = (num_reserved_tokens) as i64;
        let unknown_token = if use_unknowns {
            num_reserved_tokens + keywords.len()
        } else {
            0
        } as i64;
        let mut token_dict = HashMap::new();
        for (idx, keyword) in keywords.into_iter().enumerate() {
            token_dict.insert(keyword, idx as i64 + first_token);
        }
        Tokenizer {
            use_unknowns,
            num_reserved_tokens,
            unknown_token,
            token_dict,
        }
    }
    pub fn tokenize(&self, sentence: &str) -> Vec<Token> {
        let words = get_symbols(sentence);
        words
            .into_iter()
            .flat_map(|word| match self.token_dict.get(word) {
                None => {
                    if self.use_unknowns {
                        Some(self.unknown_token)
                    } else {
                        None
                    }
                }
                Some(tok) => Some(*tok),
            })
            .collect()
    }
    pub fn to_pickleable(self) -> PickleableTokenizer {
        (
            self.use_unknowns,
            self.num_reserved_tokens,
            self.unknown_token,
            self.token_dict,
        )
    }
    pub fn from_pickleable(tup: PickleableTokenizer) -> Self {
        Tokenizer {
            use_unknowns: tup.0,
            num_reserved_tokens: tup.1,
            unknown_token: tup.2,
            token_dict: tup.3,
        }
    }
    pub fn num_tokens(&self) -> i64 {
        (self.token_dict.len() + self.num_reserved_tokens + if self.use_unknowns { 1 } else { 0 })
            as i64
    }
    pub fn tokens(&self) -> Vec<String> {
	self.token_dict.keys().cloned().collect()
    }
}
static SYMBOLS_REGEX: &'static str = r",|:=|:>|:|=>|<=|>=|=|<>|>|<[^-]|->|<-|@@|\+{1,2}|\*{1,2}|-|~|/\\|\\/|/|%|\^|\|=|&&|\|\||\)|\(|\|\}|\{\||@\{|\{|\}|;|\|)|\{\||\|\}|\[|\]";
pub fn get_words(string: &str) -> Vec<&str> {
    lazy_static! {
        static ref WORDS: Regex =
            Regex::new(&format!(r"({}|(\??([[:word:]'!#\.])+)", SYMBOLS_REGEX)).unwrap();
    }
    WORDS.find_iter(string).map(|m| m.as_str()).collect()
}
pub fn get_symbols(string: &str) -> Vec<&str> {
    lazy_static! {
        static ref WORDS: Regex =
            Regex::new(&format!(r"({}|\.|(\??([[:word:]'!#])+)", SYMBOLS_REGEX)).unwrap();
    }
    WORDS.find_iter(string).map(|m| m.as_str()).collect()
}

pub fn normalize_sequence_length<T>(
    mut seq: Vec<T>,
    length: usize,
    pad_value: T,
) -> Vec<T>
where T: Copy {
    if seq.len() > length {
        seq.truncate(length);
    } else if seq.len() < length {
        seq.extend([pad_value].repeat(length - seq.len()));
    }
    assert_eq!(seq.len(), length);
    seq
}
pub fn normalize_sentence_length(
    tokenlist: Vec<IdentChunk>,
    length: usize,
    chunk_length: usize,
    pad_value: i64,
) -> Vec<IdentChunk> {
    let mut normalized_sentence: Vec<(i64, Vec<i64>)> =
        tokenlist.into_iter().take(length).map(
            |(key_idx, subwords)| {
                let norm_chunks = normalize_sequence_length(subwords, chunk_length, 0);
                (key_idx, norm_chunks)
            }).collect();
    while normalized_sentence.len() < length {
        normalized_sentence.push((0, vec![0; chunk_length]))
    }
    normalized_sentence.truncate(length);
    for (keyword_idx, subwords) in normalized_sentence.iter() {
    }
    normalized_sentence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_match_test() {
        let mut tokenizer = LongestMatchTokenizer::new_from_vocab(true, 2,
                                                                  vec!["some".to_string(),
                                                                       "thing".to_string()]);
        assert_eq!(
            tokenizer.tokenize("something some athing a some"),
            vec![2, 3, 2, 4, 3, 4, 2]
        );
    }
}
