def merge_columns(file1, file2, outfile_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(outfile_file, 'w') as out:
        for line1, line2 in zip(f1 , f2):
            column1 = line1.split()
            column2 = line2.split()
            
            out.write(f"{column1[0]} {column2[0]}\n")

            
file1 = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/evaluation.txt'  # Replace with your first input file path
file2 = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/ILSVRC2012_validation_ground_truth.txt'  # Replace with your second input file path
output_file = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/evaluation_out.txt'  # Replace with your desired output file path

merge_columns(file1, file2, output_file)       