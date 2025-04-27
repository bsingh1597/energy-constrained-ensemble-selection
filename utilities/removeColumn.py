def remove_column(input_file, output_file, column_index):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines:
            # Split the line into columns
            columns = line.split()
            # Remove the specified column
            if 0 <= column_index < len(columns):
                del columns[column_index]
            # Write the remaining columns back to the output file
            outfile.write(' '.join(columns) + '\n')

# Usage
input_file = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/val.txt'  # Replace with your input file path
output_file = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/evaluation.txt'  # Replace with your desired output file path
column_index = 1  # Index of the column to remove (0-based)

remove_column(input_file, output_file, column_index)


#Sequence: 
# 1. Run removeColumn.py to remove the extra column in val.text
# 2. Run mergeColumns to create a file ghaving the image & index of the image
# 3. Run createLabelIndicesMap to map the image with its type; delete the first 50,000 records not needed 
