use anyhow::Result;
use programming_assn_3::{read_vocab, read_dataset, Dataset};
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};

fn main() -> Result<()> {
    // Training data
    let mut training_file = BufReader::new(File::open("trainingSet.txt")?);
    let training_vocab = read_vocab(&mut training_file)?;
    training_file.seek(SeekFrom::Start(0))?;
    let training_data = read_dataset(&mut training_file, &training_vocab)?;

    let table = CondProbTable::from_data(&training_data, training_vocab.len());

    let feature = 1;
    for &cd in &[true, false] {
        for &x in &[true, false] {
            println!("CD={}, X={}: {}", cd, x, table.p_x_cd(cd, x, feature));
        }
    }

    Ok(())
}

/// Conditional probability table of a trained bayes net
/// For each of CD={true, false}, a table describing for each word in the vocabulary how many times it appears in the training set
#[derive(Default)]
struct CondProbTable {
    cd_false: Vec<u64>,
    cd_true: Vec<u64>,
    n_cd_true: u64,
    n_rows: usize,
}

impl CondProbTable {
    /// Construct the conditional probability table from a dataset
    pub fn from_data(data: &Dataset, vocab_len: usize) -> Self {
        let mut cd_true = vec![0; vocab_len];
        let mut cd_false = vec![0; vocab_len];
        let mut n_cd_true = 0;

        for row in data {
            let table_cd = match row.class {
                true => &mut cd_true,
                false => &mut cd_false,
            };
            for &feature in &row.features {
                table_cd[feature] += 1;
            }
            if row.class {
                n_cd_true = 0;
            }
        }

        Self {
            n_cd_true,
            cd_true,
            cd_false,
            n_rows: data.len(),
        }
    }

    /// Calculate P(CD=true)
    pub fn p_cd(&self) -> f32 {
        self.n_cd_true as f32 / self.n_rows as f32
    }

    /// Calculate P(CD=cd, X_feature=x). Returns uniform Dirichlet Prior if no rows had this class
    pub fn p_x_cd(&self, cd: bool, x: bool, feature: usize) -> f32 {
        // Select table with CD=cd
        let table = match cd {
            true => &self.cd_true,
            false => &self.cd_false,
        };

        // Number of records with X_feature=true and CD=cd
        let n_x_true = table[feature];

        // Number of records with X_feature=x and CD=cd
        let p_x_count = match x {
            true => n_x_true,
            false => self.n_rows as u64 - n_x_true,
        };

        // Number of records with CD=cd
        let p_cd_count = match cd {
            true => self.n_cd_true,
            false => self.n_rows as u64 - self.n_cd_true,
        };

        let n_j = 2.;
        if n_x_true == 0 {
            // Dirichlet prior
            1. / n_j 
        } else {
            (p_x_count as f32 + 1.) / (self.n_rows as f32 + n_j)
        }
    }
}
