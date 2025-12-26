#!/bin/bash

featureCounts_output="featureCounts_results.txt"
log_file="pipeline_$(date +"%Y%m%d%H%M%S").log"

#Add log records
exec &> >(tee -a "$log_file")

#Function: Handling error messages
handle_error() {
    local exit_code=$?
    echo "Error occurred in $0 (exit code $exit_code)" >&2
    echo "Check the log file: $log_file" >&2
    exit $exit_code
}

#Handling error signals
trap 'handle_error' ERR

#Find all *. in the current directory R1.fastq.gz file
for file in *.R1.fastq.gz; do
    basename="${file%.R1.fastq.gz}"
    file_1="${basename}.R1.fastq.gz"
    file_2="${basename}.R2.fastq.gz"
    
    #Check if the paired fastq.gz file exists
    if [ -f "$file_1" ] && [ -f "$file_2" ]; then
        echo "Processing files: $file_1 and $file_2"
        
        #Quality control and quality trimming
        fastp -i "$file_1" -I "$file_2" \
        -o "${basename}_clean.R1.fastq.gz" -O "${basename}_clean.R2.fastq.gz" \
        --adapter_sequence auto --detect_adapter_for_pe \
        --unpaired1 "${basename}_um.R1.fastq.gz" --unpaired2 "${basename}_um.R2.fastq.gz" \
        --failed_out "${basename}_failed.fastq.gz" \
        --cut_front --cut_front_window_size=1 --cut_front_mean_quality=3 \
        --cut_tail --cut_tail_window_size=1 --cut_tail_mean_quality=3 \
        --cut_right --cut_right_window_size=4 --cut_right_mean_quality=15 \
        --length_required=36 --thread 2 --trim_front1 10 --trim_front2 10 --trim_poly_x
        
        #Delete the original fastq.gz file
        [ -f "$file_1" ] && rm -f "$file_1"
        [ -f "$file_2" ] && rm -f "$file_2"
        
        #Comparison
        hisat2 -q -x /home/cht/rna-seq/genecode_data/hisat2_index/genome_tran -p 4 \
        -1 "${basename}_clean.R1.fastq.gz" -2 "${basename}_clean.R2.fastq.gz" \
        -S "hisat2_${basename}.sam" --summary-file "hisat2_${basename}.summary" &&
        
       #Convert the comparison results to BAM format and sort them
        samtools view -Sb "hisat2_${basename}.sam" | samtools sort -@ 4 -O BAM > "hisat2_${basename}.sorted.bam" &&
        
        #Delete Sam file
        [ -f "hisat2_${basename}.sam" ] && rm -f "hisat2_${basename}.sam" &&
        
        #Establish an index for BAM files
        [ -f "hisat2_${basename}.sorted.bam" ] && samtools index -@ 4 "hisat2_${basename}.sorted.bam"
    
        #Delete the cleaned fastq.gz file
        [ -f "${basename}_clean.R1.fastq.gz" ] && rm -f "${basename}_clean.R1.fastq.gz"
        [ -f "${basename}_clean.R2.fastq.gz" ] && rm -f "${basename}_clean.R2.fastq.gz"
    
        #Delete Fastp's log files
        [ -f "${basename}_failed.fastq.gz" ] && rm -f "${basename}_failed.fastq.gz"
        [ -f "${basename}_um.R1.fastq.gz" ] && rm -f "${basename}_um.R1.fastq.gz"
        [ -f "${basename}_um.R2.fastq.gz" ] && rm -f "${basename}_um.R2.fastq.gz"
    
        #Delete the summary file of hisat2
        [ -f "hisat2_${basename}.summary" ] && rm -f "hisat2_${basename}.summary"
    else
        echo "Paired files for ${basename} do not exist. Skipping this entry."
    fi
done

#Use featureCounts for gene counting and save exit status
find . -name "hisat2*.sorted.bam" | sort | xargs featureCounts -T 4 -p --countReadPairs -a /home/cht/rna-seq/genecode_data/gencode.v46.annotation.gtf -t exon -g gene_id -o "$featureCounts_output"
featureCounts_status=$?

#Check if featureCounts runs successfully
if [ "$featureCounts_status" -eq 0 ]; then
    echo "The featureCounts results are saved in: $featureCounts_output"
else
    echo "Feature Counts encountered an error while running, please check the log file: $log_file"
fi

#Output log file path
echo "The log file is saved in: $log_file"

