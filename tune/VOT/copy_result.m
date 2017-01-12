a = dir('.\results\DCFNet*');
for i = 1:numel(a)
    b = ['.\results\' a(i).name '\baseline\'];
    VOT16 = importdata('./sequences/list.txt');
    
    for j = 1:60
        c = [b VOT16{j}];
        txt_file = dir([c,'\*1.txt']);
        time_txt_file = [c,'\' VOT16{j} '_time.txt'];
        if numel(txt_file) > 0
            copyfile([c '\' txt_file.name],[c '\' txt_file.name(1:end-5) '2.txt']);
            copyfile([c '\' txt_file.name],[c '\' txt_file.name(1:end-5) '3.txt']);
            T = dlmread(time_txt_file);
            if size(T,2) == 1
                dlmwrite(time_txt_file,repmat(T,[1,3]));
            end
        end
    end
end


