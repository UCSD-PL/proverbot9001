use crate::scraped_data::NormalFloat;
use pyo3::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

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
    T: Eq + Hash + Clone,
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
}

#[pyclass]
pub struct Tokenizer {
    use_unknowns: bool,
    num_reserved_tokens: usize,
    unknown_token: Token,
    token_dict: HashMap<String, Token>,
}

pub type PickleableTokenizer = (bool, usize, Token, HashMap<String, Token>);

impl Tokenizer {
    pub fn new(use_unknowns: bool, num_reserved_tokens: usize, keywords_filepath: String) -> Self {
        println!("Getting relevant keywords");
        let keywords = relevant_keywords(data, num_keywords);
        let first_token = (num_reserved_tokens + if use_unknowns { 1 } else { 0 }) as i64;
        let unknown_token = if use_unknowns { num_reserved_tokens } else { 0 } as i64;
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
        let words = get_words(sentence);
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
}

fn entropy<T>(outputs: impl IntoIterator<Item = T>) -> f64
where
    T: Hash + Eq,
{
    let mut counts = HashMap::new();
    let mut total_counts = 0;
    for output in outputs.into_iter() {
        total_counts += 1;
        *counts.entry(output).or_insert(0) += 1;
    }
    let mut result: f64 = 0.0;
    for (_output, count) in counts.into_iter() {
        let probability = (count as f64) / (total_counts as f64);
        result += -(probability * probability.log2());
    }
    return result;
}

fn multipartition<T, K, F>(xs: impl IntoIterator<Item = T>, f: F) -> Vec<Vec<T>>
where
    T: Clone,
    F: Fn(T) -> K,
    K: Hash + Eq,
{
    let mut partitions = HashMap::new();
    for x in xs.into_iter() {
        let k = f(x.clone());
        (*partitions.entry(k).or_insert(Vec::new())).push(x);
    }
    partitions.into_iter().map(|(_k, v)| v).collect()
}

fn count<T>(xs: impl IntoIterator<Item = T>) -> HashMap<T, usize>
where
    T: Hash + Eq,
{
    let mut counts = HashMap::new();
    for x in xs.into_iter() {
        *counts.entry(x).or_insert(0) += 1;
    }
    counts
}

fn top_n<T>(scores: &HashMap<T, usize>, n: usize) -> Vec<T>
where
    T: Ord + Hash + Eq + Clone,
{
    let mut heap: BinaryHeap<(usize, T)> = scores
        .iter()
        // Put the scores first so they dominate the Ord implementation on pairs
        .map(|(item, score)| (score.clone(), item.clone()))
        .collect();
    let mut result = Vec::new();
    for _ in 0..n {
        match heap.pop() {
            Some(v) => result.push(v.1),
            None => break,
        }
    }
    result
}

pub fn get_words(string: &str) -> Vec<&str> {
    lazy_static! {
        static ref WORDS: Regex = Regex::new(r"(,|\.+|:\b|:=|\)|\()|(([[:word:]]|')+)").unwrap();
    }
    WORDS.find_iter(string).map(|m| m.as_str()).collect()
}

fn word_partitioned_entropy(pairs: &Vec<(&String, usize)>, word: &str) -> f64 {
    let num_pairs = pairs.len();
    let partitions = multipartition(pairs, |(string, _tactic)| {
        get_words(&string).contains(&word)
    });
    let mut scaled_partition_entropies_iter = partitions
        .into_iter()
        .map(|part| (part.len() as f64) * entropy(part));
    (scaled_partition_entropies_iter.next().unwrap()
        + scaled_partition_entropies_iter.next().unwrap())
        / 2
}

fn relevant_keywords(pairs: Vec<(String, usize)>, num_keywords: usize) -> Vec<String> {
    fn leader_entropy(pool: &Vec<(String, usize)>) -> (usize, f64) {
        if pool.len() == 0 {
            return (0, 0.0);
        }
        let tactic_counts = count(pool.iter().map(|(_string, tactic)| tactic));
        let (leader_tactic, _leader_count) = tactic_counts
            .into_iter()
            .max_by_key(|(_tactic, count)| count.clone())
            // This unwrap should be safe because of the branch above on the size of pool.
            .unwrap();
        (
            *leader_tactic,
            entropy(
                pool.into_iter()
                    .map(|(_string, output)| if *output == *leader_tactic { 1 } else { 0 }),
            ),
        )
    }
    // Given a pools list, split each pool into two pools based on the
    // presence of the word 'word' in the samples, dropping pools with
    // no entropy (only one tactic).
    fn split_pools(
        pools: Vec<(Vec<(String, usize)>, usize, f64)>,
        word: &str,
    ) -> Vec<(Vec<(String, usize)>, usize, f64)> {
        let mut new_pools = Vec::new();
        for (old_pool, _old_leader, _old_entropy) in pools.into_iter() {
            let subpools = multipartition(old_pool, |(string, _tactic)| {
                if get_words(&string).into_iter().any(|w| w == word) {
                    1
                } else {
                    0
                }
            });
            for subpool in subpools.into_iter() {
                if subpool.len() > 1 {
                    let (leader, entropy) = leader_entropy(&subpool);
                    new_pools.push((subpool, leader, entropy));
                }
            }
        }
        new_pools
    }
    // Count each of the words
    let word_count = count(pairs.iter().flat_map(|(string, _tactic)| get_words(string)));
    // Our candidate pool will be the n**2 most common words
    let mut common_words = top_n(&word_count, num_keywords.pow(2));

    let (total_leader, total_leader_entropy) = leader_entropy(&pairs);
    let mut pools = vec![(pairs.clone(), total_leader, total_leader_entropy)];
    let mut keywords = Vec::new();

    // The first 25% of the keywords are made up of the most
    // common words, since those are likely to be useful
    // syntactically, but not register as high entropy.
    let really_common_words = top_n(&word_count, num_keywords / 4);
    for word in really_common_words.into_iter() {
        common_words.remove_item(&word);
        keywords.push(word);
    }

    while keywords.len() < num_keywords && pools.len() > 0 && common_words.len() > 0 {
        // This unwrap is safe because we checked pool.len() in
        // the loop condition.
        let (idx, highest_entropy_pool_entry) = pools
            .iter()
            .enumerate()
            .max_by_key(|(idx, (_pool, _leader, entropy))| {
                (NormalFloat::new(*entropy), idx.clone())
            })
            .unwrap();
        let (highest_entropy_pool, leader, pool_entropy) = highest_entropy_pool_entry;
        println!("Entropy of pool is {}, leader is {}", pool_entropy, leader);

        let leader_info = highest_entropy_pool
            .iter()
            .map(|(string, tactic)| (string, if tactic == leader { 1 } else { 0 }))
            .collect();
        println!("getting word entropy pairs");
        let word_part_entropy_pairs: Vec<(&str, f64)> = common_words
            .iter()
            .map(|word| (&word[..], word_partitioned_entropy(&leader_info, word)))
            .collect();
        // This unwrap is safe because word_entropy_pairs is the
        // same size as common_words, which we checked was greater
        // than 0 in the loop condition.
        let (word, part_entropy) = word_part_entropy_pairs
            .into_iter()
            .max_by_key(|(_word, part_entropy)| NormalFloat::new(*part_entropy))
            .unwrap();
        println!("Word is {}", word);
        if part_entropy >= *pool_entropy {
            pools.remove(idx);
            println!("{} pools left", pools.len());
            continue;
        }
        println!("{} keywords picked", keywords.len());
        keywords.push(word);
        common_words.remove_item(&word);
        pools = split_pools(pools, word);
    }
    keywords.into_iter().map(str::to_owned).collect()
}
