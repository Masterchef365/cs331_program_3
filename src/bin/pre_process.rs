use anyhow::Result;
use programming_assn_3::{read_vocab, read_dataset};
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};

fn main() -> Result<()> {
    let mut training_file = BufReader::new(File::open("trainingSet.txt")?);

    let training_vocab = read_vocab(&mut training_file)?;
    dbg!(training_vocab);

    training_file.seek(SeekFrom::Start(0))?;
    //let training_vocab = read_dataset(&mut training_file, &training_vocab)?;

    Ok(())
}
