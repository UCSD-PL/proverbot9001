use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::iter;
use pyo3::prelude::*;

pub type LongTensor2D = Vec<Vec<i64>>;
pub type FloatTensor2D = Vec<Vec<f64>>;

pub type LongTensor1D = Vec<i64>;
pub type FloatTensor1D = Vec<f64>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScrapedTactic {
    pub relevant_lemmas: Vec<String>,
    pub prev_tactics: Vec<String>,
    pub prev_hyps: Vec<String>,
    pub prev_goal: String,
    pub tactic: String,
}

pub struct VernacCommand {
    pub command: String,
}

pub enum ScrapedData {
    Vernac(VernacCommand),
    Tactic(ScrapedTactic),
}

pub fn scraped_from_file(file: File) -> impl iter::Iterator<Item = ScrapedData> {
    BufReader::new(file).lines().map(|line: Result<String>| {
        let actual_line = line.expect("Couldn't read line");
        if actual_line.starts_with("\"") {
            ScrapedData::Vernac(VernacCommand {
                command: serde_json::from_str(&actual_line).expect("Couldn't parse string"),
            })
        } else {
            ScrapedData::Tactic(serde_json::from_str(&actual_line).expect("Couldn't parse line"))
        }
    })
}


#[pyclass]
#[derive(Default, Clone)]
pub struct DataloaderArgs {
    #[pyo3(get, set)]
    pub max_distance: usize,
    #[pyo3(get, set)]
    pub max_string_distance: usize,
    #[pyo3(get, set)]
    pub max_length: usize,
    #[pyo3(get, set)]
    pub num_keywords: usize,
}
#[pymethods]
impl DataloaderArgs {
    #[new]
    fn new(obj: &PyRawObject) {
        let d: DataloaderArgs = Default::default();
        obj.init({d});
    }
}
impl<'source> pyo3::FromPyObject<'source> for DataloaderArgs {
    fn extract(ob: &'source pyo3::types::PyAny) -> pyo3::PyResult<DataloaderArgs> {
        let cls: &DataloaderArgs = pyo3::PyTryFrom::try_from(ob)?;
        Ok(cls.clone())
    }
}

pub struct NormalFloat(f64);
impl NormalFloat {
    pub fn new(v: f64) -> NormalFloat {
        assert_eq!(v, v);
        NormalFloat(v)
    }
}
impl PartialEq for NormalFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for NormalFloat {}
impl PartialOrd for NormalFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.0 > other.0 {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Less)
        }
    }
}
impl Ord for NormalFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
