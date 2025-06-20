# File paths (input and output files)
input_file = 'requirements01.txt'
output_file = 'cleaned_requirements.txt'

# Open the input file and create the output file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Remove everything after (and including) the '@'
        clean_line = line.split('@')[0].strip()
        if clean_line:  # Avoid blank lines
            outfile.write(clean_line + '\n')

print(f"Cleaned file saved as '{output_file}'")
