import ID_verify
import extraction

# Input file: input.csv
# - Should have a column named "screenshots"
# Output file: verified_transactions.csv

if __name__ == "__main__":
    print("Starting extraction of transaction id...")
    extraction.main()
    print("Starting ID verification...")
    ID_verify.main()
