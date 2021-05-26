# Programming assignment 3
Before you run any of these programs, ensure that you have `trainingSet.txt` and `testSet.txt` in the root of this project (I.E. alongside `Cargo.toml`).

To run just the preprocessing step, execute the following:
```sh
cargo run --release --bin pre_process
```
This should generate `preprocessed_test.txt` and `preprocessed_train.txt`. Note that these files are **not** used in the training/inference step. They are seperate code because I use a sparse representation for inference, which is much, much more efficient.

To run the training and evaluation step, execute the following:
```sh
cargo run --release --bin classify
```
