use anyhow::Result;
use programming_assn_3::{read_vocab, read_dataset, Vocabulary, Dataset};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write, Seek, SeekFrom};

fn main() -> Result<()> {
    // Training data
    let mut training_file = BufReader::new(File::open("trainingSet.txt")?);
    let training_vocab = read_vocab(&mut training_file)?;

    training_file.seek(SeekFrom::Start(0))?;
    let training_data = read_dataset(&mut training_file, &training_vocab)?;

    let mut preprocessed_train = BufWriter::new(File::create("preprocessed_train.txt")?);
    write_features(&mut preprocessed_train, &training_data, &training_vocab)?;
    drop((training_data, training_file));

    // Test data
    let mut test_file = BufReader::new(File::open("testSet.txt")?);
    let test_data = read_dataset(&mut test_file, &training_vocab)?;

    let mut preprocessed_test = BufWriter::new(File::create("preprocessed_test.txt")?);
    write_features(&mut preprocessed_test, &test_data, &training_vocab)?;

    Ok(())
}

fn write_features(file: &mut impl Write, dataset: &Dataset, vocab: &Vocabulary) -> Result<()> {
    // Write vocab
    for word in vocab {
        write!(file, "{},", word)?;
    }
    writeln!(file)?;

    // Write rows - essentially converting to a dense representation from a sparse one
    for row in dataset {
        let mut pos = 0;

        // Write features with zeroes in between
        for &feature in &row.features {
            while pos != feature {
                write!(file, "0,")?;
                pos += 1;
            }
            write!(file, "1,")?;
            pos += 1;
        }

        // Write remaining zeroes
        while pos < vocab.len() {
            write!(file, "0,")?;
            pos += 1;
        }

        if row.class {
            write!(file, "1")?; 
        } else {
            write!(file, "0")?; 
        }
        writeln!(file)?;
    }

    Ok(())
}
