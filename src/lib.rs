use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::io::BufRead;

/// A vocabulary consists of all words in alphabetical order
pub type Vocabulary = Vec<String>;

/// A dataset consists of a set of rows
pub type Dataset = Vec<Row>;

/// A row consists of a set of indices into an associated alphabet
#[derive(Debug, Clone)]
pub struct Row {
    /// Set of indices into an associated alphabet, excluding those not present in the alphabet
    pub features: Vec<usize>,
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
        for s in parts {
            let word = prepare_word(s);
            if !word.is_empty() {
                vocab.insert(word);
            }
        }
    }

    // Convert to sorted array
    let mut vocab: Vec<String> = vocab.into_iter().collect();
    vocab.sort_unstable(); // Note that this sorts by unicode codepoint, not alphabetically in other languages!

    Ok(vocab)
}

/// Read the dataset into a set of rows, using the vocab for feature indices
pub fn read_dataset(reader: &mut impl BufRead, vocab: &[String]) -> Result<Dataset> {
    let mut dataset = vec![];

    for (line_idx, line) in reader.lines().enumerate() {
        // Split line into parts
        let line = line?;
        let mut parts: Vec<&str> = line.split_whitespace().collect();

        // Parse class
        let missing_class = || format!("Line {} missing class", line_idx + 1);
        let class = parts.pop().with_context(missing_class)?;
        let class = match class {
            "1" => true,
            "0" => false,
            _ => bail!("{}", missing_class()),
        };

        // Segment words and add them to the vocab, sorting them in order
        let mut features = vec![];
        for word in parts {
            let word = prepare_word(word);
            if word.is_empty() {
                continue;
            }
            match vocab.binary_search(&word) {
                Ok(idx) => features.push(idx),
                Err(_) => continue, // Note: We skip words not in the vocab!
            };
        }

        // Sort and dedup for later searching
        features.sort_unstable();
        features.dedup();

        dataset.push(Row { features, class });
    }

    Ok(dataset)
}

/// Prepare a word
pub fn prepare_word(word: &str) -> String {
    word.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect()
}
