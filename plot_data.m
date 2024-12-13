clear all;  clc;
% close all;
% .mat 파일 로드
file_date = '2024-11-15';
channel = '/awgn/';
latency = '0ms';

channel_param = append(channel,latency);
for i = 1:1:1
    idx_path = append(channel_param, strcat('/',int2str(i)));
    file_path = append(file_date,idx_path);
    data_path = append(file_path,'/data.mat');
    param_path = append(file_path,'/Hyper_Param.txt');
    param = readcell(param_path);
    
    main_param = struct();
    
    PARAM={'channel_type','SNR', 'comm_latency'};
    for p = PARAM
        idx = find(strcmp(param(:, 1), p)); 
        value = param{idx, 2}
        main_param.(p{1}) = value;
    end
    
    
    data = load(data_path);
    
    window_size = 500;
    lifting_time = data.stable_lifting_time;
    avg_lifting_time = movmean(lifting_time, window_size);
    box_pos = data.box_z_pos;
    avg_box_pos = movmean(box_pos, window_size);
    
    
    legendText = sprintf('%d ms', main_param.comm_latency);
    if main_param.channel_type == "awgn"
        titleText = sprintf('AWGN(SNR=%d dB)',main_param.SNR);
    else
        titleText = sprintf('Fading(SNR=%d dB)',main_param.SNR);
    end
    
    
    
    
    figure();
    plot(avg_lifting_time,'k','LineWidth',1.5); hold on;
    ylim([0 680]);
    
    legend(legendText','Interpreter','latex');
    grid on;
    xlabel('Episode','FontSize',14,'Interpreter','latex');
    ylabel('Time holding the object stable','FontSize',14,'Interpreter','latex');
    title(titleText,'FontSize',14,'Interpreter','latex');
    
    figure();
    plot(avg_box_pos,'k','LineWidth',1.5); hold on;
    legend(legendText','Interpreter','latex');
    grid on;
    ylim([0,2.8]);
    xlabel('Episode','FontSize',14,'Interpreter','latex');
    ylabel('Box z-pos','FontSize',14,'Interpreter','latex');
    title(titleText,'FontSize',14,'Interpreter','latex');
end