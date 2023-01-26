use pyo3::prelude::*;
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
pub type Path = i64;

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
    pub fn lookup(&mut self, v: T) -> i64 {
        if !self.map.contains_key(&v) {
            if self.frozen {
                return 0;
            } else {
                self.map.insert(v.clone(), self.next_idx);
                self.next_idx += 1;
            }
        }
        *self.map.get(&v).unwrap()
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
    pub fn load_from_text(path: &str) -> OpenIndexer::<String> {
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

#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct Tokenizer {
    use_unknowns: bool,
    num_reserved_tokens: usize,
    unknown_token: Token,
    token_dict: HashMap<String, Token>,
    paths_dict: HashMap<String, Path>,
}

pub type PickleableTokenizer = (bool, usize, Token, HashMap<String, Token>, HashMap<String, Token>);

impl Tokenizer {
    pub fn new(use_unknowns: bool, num_reserved_tokens: usize, keywords_filepath: &str, paths_filepath: &str) -> Self {
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
        let paths: Vec<_> = io::BufReader::new(File::open("/home/zhannakaufma_umass_edu/work/proverbot9001/common_paths.txt").expect(&format!(
            "Couldn't open paths file \"{}\"",
            paths_filepath
        )))
        .lines()
        .map(|path| path.unwrap())
        .collect();
        let mut paths_dict = HashMap::new();
        for (idx, path) in paths.into_iter().enumerate() {
            paths_dict.insert(path, idx as i64 + first_token);
        }
        Tokenizer {
            use_unknowns,
            num_reserved_tokens,
            unknown_token,
            token_dict,
            paths_dict,
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
    pub fn pathstokenize(&self, sentence: &str) -> (Vec<Token>, Vec<Path>) {
        let wordsplit = sentence.split(' ');
        let wordsplit_two = sentence.split(' ');
        let words = wordsplit.collect::<Vec<&str>>();
        let words_two = wordsplit_two.collect::<Vec<&str>>();


        let goalstokens: Vec<Token> = words
            .into_iter()
            .flat_map(|word| match self.token_dict.get(word.split("|-path-|").collect::<Vec<&str>>()[0]) {
                None => {
                    if self.use_unknowns {
                        Some(self.unknown_token)
                    } else {
                        None
                    }
                }
                Some(tok) => Some(*tok),
            })
            .collect();

        let path_lookup_closure = |word: &str| -> Option<Path> {
            let word_split = word.split("|-path-|").collect::<Vec<&str>>();
            if word_split.len() == 1 {
                Some((self.paths_dict.len() as i64))
            }
            else {
                match self.paths_dict.get(word_split[1]) {
                    None => Some((self.paths_dict.len() as i64) + 1),
                    Some(tok) => Some(*tok),
                }
            }
        };

        let pathstokens: Vec<Path> = words_two
            .into_iter()
            .flat_map(path_lookup_closure)
            .collect();

        assert_eq!(goalstokens.len(), pathstokens.len());
        (goalstokens, pathstokens)
     }

    pub fn to_pickleable(self) -> PickleableTokenizer {
        (
            self.use_unknowns,
            self.num_reserved_tokens,
            self.unknown_token,
            self.token_dict,
            self.paths_dict,
        )
    }
    pub fn from_pickleable(tup: PickleableTokenizer) -> Self {
        Tokenizer {
            use_unknowns: tup.0,
            num_reserved_tokens: tup.1,
            unknown_token: tup.2,
            token_dict: tup.3,
            paths_dict: tup.4,
        }
    }
    pub fn num_tokens(&self) -> i64 {
        (self.token_dict.len() + self.num_reserved_tokens + if self.use_unknowns { 1 } else { 0 })
            as i64
    }
    pub fn num_paths_tokens(&self) -> i64 {
        (self.paths_dict.len() + 2)
            as i64
    }
    pub fn tokens(&self) -> Vec<String> {
	self.token_dict.keys().cloned().collect()
    }
    pub fn paths_tokens(&self) -> Vec<String> {
	self.paths_dict.keys().cloned().collect()
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

pub fn normalize_sentence_length(
    mut tokenlist: Vec<i64>,
    length: usize,
    pad_value: i64,
) -> Vec<i64> {
    if tokenlist.len() > length {
        tokenlist.truncate(length);
    } else if tokenlist.len() < length {
        tokenlist.extend([pad_value].repeat(length - tokenlist.len()));
    }
    tokenlist
}
