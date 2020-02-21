use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::iter;

// enum PyTensor {
//     Long2DTensor(Vec<Vec<i64>>),
//     Float2DTensor(Vec<Vec<f64>>),
// }

#[derive(Debug, Serialize, Deserialize)]
struct ScrapedTactic {
    relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    prev_hyps: Vec<String>,
    prev_goal: String,
    tactic: String,
}

struct VernacCommand {
    command : String,
}

enum ScrapedData {
    Vernac(VernacCommand),
    Tactic(ScrapedTactic),
}

type PyTensor = Vec<Vec<i64>>;

#[pyfunction]
fn load_tactics(filename: String) -> PyResult<Option<Vec<PyTensor>>> {
    println!("Reading dataset.");
    match File::open(filename) {
        Result::Ok(file) => {
            for point in scraped_from_file(file) {
                match point {
                    ScrapedData::Vernac(_cmd) => (),
                    ScrapedData::Tactic(tactic) => {
                        println!("{:?}", tactic);
                        break;
                    }
                }
            }
            Ok(None)
        }
        Result::Err(err) => {
            Err(PyErr::new::<exc::TypeError, _>("Failed to open file"));
        }
    }
}
fn scraped_from_file(file: File) -> impl iter::Iterator<Item=ScrapedData> {
    BufReader::new(file).lines().map(|line : Result<String>| {
        let actual_line = line.expect("Couldn't read line");
        if actual_line.starts_with("\""){
            ScrapedData::Vernac(VernacCommand{command: serde_json::from_str(&actual_line).expect("Couldn't parse string")})
        } else {
            ScrapedData::Tactic(serde_json::from_str(&actual_line).expect("Couldn't parse line"))
        }
    })
}
#[pymodule]
fn dataloader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(load_tactics))?;
    Ok(())
}
