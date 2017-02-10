function main_running_DCFNet(file,param,evalType)
warning off all;

addpath('./util');

addpath(('./rstEval'));

seqs=configSeqs('otb2013');

trackers={struct('name','DCFNet','namePaper','DCFNet')};

shiftTypeSet = {'left','right','up','down','topLeft','topRight','bottomLeft','bottomRight','scale_8','scale_9','scale_11','scale_12'};

diary(['./tmp/' evalType '.txt']);

numSeq=length(seqs);
numTrk=length(trackers);

finalPath = [file '/'];

if ~exist(finalPath,'dir')
    mkdir(finalPath);
end

tmpRes_path = ['./tmp/' evalType '/'];
bSaveImage=0;

if ~exist(tmpRes_path,'dir')
    mkdir(tmpRes_path);
end

pathAnno = './anno/';

for idxSeq=1:length(seqs)
    s = seqs{idxSeq};
    
%      if ~strcmp(s.name, 'coke')
%         continue;
%      end
        
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,'\img\',id,'.',s.ext);
    end
    
    img = imread(s.s_frames{1});
    [imgH,imgW,ch]=size(img);
    
    rect_anno = dlmread([pathAnno s.name '.txt']);
    numSeg = 20;
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    
    switch evalType
        case 'SRE'
            subS = subSeqs{1};
            subA = subAnno{1};
            subSeqs=[];
            subAnno=[];
            r=subS.init_rect;
            
            for i=1:length(shiftTypeSet)
                subSeqs{i} = subS;
                shiftType = shiftTypeSet{i};
                subSeqs{i}.init_rect=shiftInitBB(subS.init_rect,shiftType,imgH,imgW);
                subSeqs{i}.shiftType = shiftType;
                
                subAnno{i} = subA;
            end

        case 'OPE'
            subS = subSeqs{1};
            subSeqs=[];
            subSeqs{1} = subS;
            
            subA = subAnno{1};
            subAnno=[];
            subAnno{1} = subA;
        otherwise
    end

            
    for idxTrk=1:numTrk
        t = trackers{idxTrk};

%         if ~strcmp(t.name, 'LSK')
%             continue;
%         end

        % validate the results
        if exist([finalPath s.name '_' t.name '.mat'])
            load([finalPath s.name '_' t.name '.mat']);
            bfail=checkResult(results, subAnno);
            if bfail
                disp([s.name ' '  t.name]);
            end
            continue;
        end

        switch t.name
            case {'VTD','VTS'}
                continue;
        end

        results = [];
        for idx=1:length(subSeqs)
            disp([num2str(idxTrk) '_' t.name ', ' num2str(idxSeq) '_' s.name ': ' num2str(idx) '/' num2str(length(subSeqs))])       

            rp = [tmpRes_path s.name '_' t.name '_' num2str(idx) '/'];
            if bSaveImage&~exist(rp,'dir')
                mkdir(rp);
            end
            
            subS = subSeqs{idx};
            
            subS.name = [subS.name '_' num2str(idx)];
            
%             subS.s_frames = subS.s_frames(1:20);
%             subS.len=20;
%             subS.endFrame=subS.startFrame+subS.len-1;
            
            funcName = ['res=run_' t.name '(subS, rp, bSaveImage, param);'];

            try
                switch t.name
                    case {'VR','TM','RS','PD','MS'}
                    otherwise
                        cd(['./trackers/' t.name]);
                        addpath(genpath('./'))
                end
                
                eval(funcName);
                
                switch t.name
                    case {'VR','TM','RS','PD','MS'}
                    otherwise
                        rmpath(genpath('./'))
                        cd('../../');
                end
                
                if isempty(res)
                    results = [];
                    break;
                end
            catch err
                disp('error');
                rmpath(genpath('./'))
                cd('../../');
                res=[];
                continue;
            end
            
            res.len = subS.len;
            res.annoBegin = subS.annoBegin;
            res.startFrame = subS.startFrame;
                    
            switch evalType
                case 'SRE'
                    res.shiftType = shiftTypeSet{idx};
            end
            
            results{idx} = res;
            
        end
        save([finalPath s.name '_' t.name '.mat'], 'results');
    end
end

t=clock;
t=uint8(t(2:end));
disp([num2str(t(1)) '/' num2str(t(2)) ' ' num2str(t(3)) ':' num2str(t(4)) ':' num2str(t(5))]);

