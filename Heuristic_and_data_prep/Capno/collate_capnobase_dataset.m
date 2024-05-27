function collate_capnobase_dataset
% COLLATE_CAPNOBASE_DATASET  Collates individual subjects' data files into 
% a single MATLAB file for analysis.
%   
%  Inputs:
%
%    files    -  a MATLAB file for each individual subject in the CapnoBase 
%                dataset. Details of how to obtain these MATLAB files are 
%                provided at: https://peterhcharlton.github.io/info/datasets
%
%  Outputs:
%
%    files    -  a single MATLAB file for the whole dataset
%
%  Preparation:
%
%    Modify the MATLAB script by inserting the 'up.paths.root_folder' and 
%    'up.paths.save_folder' into the setup_up function.
%
%  Further information: 
%
%   https://peterhcharlton.github.io/info/datasets
%
%  Licence:
%       Available under the MIT License - please see the accompanying
%       file named "MIT_LICENSE.txt"
%
% Author: Peter H. Charlton, May 2021.

fprintf('\n ~~~ Collating CapnoBase dataset ~~~')

% These universal parameters will need adjusting for your computer:
up = setup_up;

collate_data(up);

% The file for analysis will now be saved at:
sprintf('%s', ['File saved at: ' up.paths.root_folder ])

end

function up = setup_up

fprintf('\n - Setting up parameters')

% The following root folder contains subfolders for each subject (name
% 'S#' where # is the subject number)
up.paths.root_folder = '.../CAPNO/mat/';
up.paths.save_folder = '.../CAPNO/mat/';

end

function collate_data(up)

fprintf('\n - Importing data');

% Identify individual recordings
files = dir(up.paths.root_folder);
filenames = extractfield(files, 'name');
filenames = filenames(cellfun(@length, filenames)>2); filenames = filenames(:);

% collate each subject's data into a single file
for subj_el = 1:length(filenames)
    
    fprintf('\n   - subject %d', subj_el);
    
    % current filename
    filename = filenames{subj_el};
    
    % setup
    SID = str2num(filename(1:4));
    if exist('data', 'var')
        struct_el = length(data)+1;
    else
        struct_el = 1;
    end
    
    % load relevant data
    rel_data = load([up.paths.root_folder, filename]);
    
    % Insert fixed params
    data(struct_el).fix.id = SID;
    data(struct_el).fix.ventilation = rel_data.meta.treatment.ventilation;
    data(struct_el).fix.age = rel_data.meta.subject.age;
    data(struct_el).fix.weight = rel_data.meta.subject.weight;
    
    % Identify group
    rel_group = [];
    if data(struct_el).fix.age >= 18
        rel_group = [rel_group, 'adult'];
    else
        rel_group = [rel_group, 'paed'];
    end
    if strcmp(data(struct_el).fix.ventilation, 'spontaneous')
        rel_group = [rel_group, '_spont'];
    else
        rel_group = [rel_group, '_vent'];
    end
    data(struct_el).group = rel_group; clear rel_group
    
    % insert PPG signal
    data(struct_el).ppg.v = rel_data.signal.pleth.y;
    data(struct_el).ppg.fs = rel_data.param.samplingrate.pleth;
    
    % insert EKG signal
    data(struct_el).ekg.v = rel_data.signal.ecg.y;
    data(struct_el).ekg.fs = rel_data.param.samplingrate.ecg;
    data(struct_el).ekg.pk = rel_data.labels.ecg.peak.x;
    data(struct_el).ekg.artif = rel_data.labels.ecg.artif.x;
    
    % insert CO2 signal
    data(struct_el).co2.v = rel_data.signal.co2.y;
    data(struct_el).co2.fs = rel_data.param.samplingrate.co2;
    
    % insert breaths
    data(struct_el).ref.breaths.t = rel_data.labels.co2.startinsp.x./rel_data.param.samplingrate.co2;
    data(struct_el).ref.breaths.units = 's';
    
    % insert hr
    data(struct_el).ref.params.hr.v = rel_data.reference.hr.ecg.y;
    data(struct_el).ref.params.hr.t = rel_data.reference.hr.ecg.x;
    data(struct_el).ref.params.hr.method = 'ecg-derived';
    data(struct_el).ref.params.hr.units.v = rel_data.reference.units.hr.y;
    data(struct_el).ref.params.hr.units.t = rel_data.reference.units.x;
    % insert pr
    data(struct_el).ref.params.pr.v = rel_data.reference.hr.pleth.y;
    data(struct_el).ref.params.pr.t = rel_data.reference.hr.pleth.x;
    data(struct_el).ref.params.pr.method = 'ppg-derived';
    data(struct_el).ref.params.pr.units.v = rel_data.reference.units.hr.y;
    data(struct_el).ref.params.pr.units.t = rel_data.reference.units.x;
    % insert rr
    data(struct_el).ref.params.rr.v = rel_data.reference.rr.co2.y;
    data(struct_el).ref.params.rr.t = rel_data.reference.rr.co2.x;
    data(struct_el).ref.params.rr.method = 'co2-derived';
    data(struct_el).ref.params.rr.units.v = rel_data.reference.units.rr.y;
    data(struct_el).ref.params.rr.units.t = rel_data.reference.units.x;
    
    clear rel_data SID filename
    
end

% Save to file
fprintf('\n   - Saving CapnoBase dataset')
filepath = [up.paths.save_folder, 'capnobase_data'];
save(filepath, 'data')

fprintf('\n\n - NB: this dataset also contains additional variables such as PPG peak annotations which have not been imported.');

fprintf('\n\n ~~~ DONE ~~~')
end
