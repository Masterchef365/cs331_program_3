use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::io::{BufRead, Read};

/// A vocabulary consists of all words in alphabetical order
pub type Vocabulary = Vec<String>;

/// A dataset consists of a set of rows
pub type Dataset = Vec<Row>;

/// A row consists of a set of indices into an associated alphabet
pub struct Row {
    /// Set of indices into an associated alphabet, excluding those not present in the alphabet
    pub words: Vec<usize>,
    /// True if the class for this row was "1", and false if it was "0"
    pub class: bool,
}

/// Construct a vocabulary from a file
pub fn read_vocab(reader: &mut impl BufRead) -> Result<Vocabulary> {
    let mut vocab = HashSet::new();

    for line in reader.lines() {
        // Split line into parts
        let line = line?;
        let mut parts: Vec<&str> = line.split_whitespace().collect();

        // Ignore class
        parts.pop();

        // Segment words and add them to the vocab
        for s in line.split_whitespace() {
            vocab.insert(prepare_word(s));
        }
    }

    // Convert to sorted array
    let mut vocab: Vec<String> = vocab.into_iter().collect();
    vocab.sort_unstable(); // Note that this sorts by unicode codepoint, not alphabetically in other languages!

    Ok(vocab)
}

/// Read the dataset, assigning
pub fn read_dataset(reader: &mut impl BufRead, vocab: &[String]) -> Result<Dataset> {
    /*
    for (line_idx, line) in reader.lines().enumerate() {
        // Parse class
        let missing_class = || format!("Line {} missing class", line_idx+1);
        let class = parts.pop().with_context(missing_class)?;
        let class = match *class {
            "1" => true,
            "0" => false,
            _ => bail!("{}", missing_class()),
        };
    */

    todo!()
}

pub fn prepare_word(word: &str) -> String {
    word.chars().filter(char::is_ascii_alphanumeric).collect()
}
