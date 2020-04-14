input_dir = '/home/lucas/datasets/pose-data/ply-models';
output_dir = '/home/lucas/datasets/pose-data/ply-models-ascii';
if ~exist(output_dir)
    mkdir(output_dir)
end
files_metadata = dir([input_dir filesep '*.ply']);
for j = 1:length(files_metadata)
    fname_in = [input_dir filesep files_metadata(j).name];
    fname_out = [output_dir filesep files_metadata(j).name];
    model = pcread(fname_in);
    pcwrite(model, fname_out, 'Encoding', 'ascii');
end
