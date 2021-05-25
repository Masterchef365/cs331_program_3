use anyhow::Result;
use programming_assn_3::{read_dataset, read_vocab, Dataset};
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};

fn main() -> Result<()> {
    // Training data
    let mut training_file = BufReader::new(File::open("trainingSet.txt")?);
    let training_vocab = read_vocab(&mut training_file)?;
    training_file.seek(SeekFrom::Start(0))?;
    let training_data = read_dataset(&mut training_file, &training_vocab)?;

    let table = CondProbTable::from_data(&training_data, training_vocab.len());

    // Evaluate training set
    let mut n_correct = 0;
    for row in &training_data {
        let prediction = infer(&row.features, &table, training_vocab.len());
        if prediction == row.class {
            n_correct += 1;
        }
    }
    println!("Training set accuracy: {}", n_correct as f32 / training_data.len() as f32);

    Ok(())
}

fn infer(true_features: &[usize], table: &CondProbTable, vocab_len: usize) -> bool {
    log_sum(true_features, true, table, vocab_len)
        >= log_sum(true_features, false, table, vocab_len)
}

fn log_sum(true_features: &[usize], cd: bool, table: &CondProbTable, vocab_len: usize) -> f32 {
    let mut sum = table.p_cd(cd).ln();
    let mut pos = 0;

    // Sum sparse features
    for &feature in true_features {
        // False features in between true features
        while pos != feature {
            sum += table.p_x_cd(cd, false, pos).ln();
            pos += 1;
        }

        // True features
        sum += table.p_x_cd(cd, true, feature).ln();
        pos += 1;
    }

    // Sum remaining false features
    while pos < vocab_len {
        sum += table.p_x_cd(cd, false, pos).ln();
        pos += 1;
    }

    sum
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
    pub fn p_cd(&self, cd: bool) -> f32 {
        self.p_cd_count(cd) as f32 / self.n_rows as f32
    }

    /// Number of records with CD=cd
    fn p_cd_count(&self, cd: bool) -> u64 {
        match cd {
            true => self.n_cd_true,
            false => self.n_rows as u64 - self.n_cd_true,
        }
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

        let n_j = 2.;
        if n_x_true == 0 {
            // Dirichlet prior
            1. / n_j
        } else {
            (p_x_count as f32 + 1.) / (self.p_cd_count(cd) as f32 + n_j)
        }
    }
}
