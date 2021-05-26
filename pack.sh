mkdir hw_3
cp -r src Cargo.toml README.md results.txt hw_3/
zip -r hw_3.zip hw_3/
rm -r hw_3/

scp hw_3.zip freemadu@flip.engr.oregonstate.edu:Projects/cs331/
#rm -r hw_3.zip
