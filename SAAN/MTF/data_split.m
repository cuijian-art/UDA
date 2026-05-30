function New_data = data_split(data, sample_len,div_num)
    [m, ~] = size(data);
    win = fix(m/sample_len);
    New_data=[];
    if win>=div_num
        for i = 1:div_num 
            split_data = data(((i-1)*sample_len)+1:i*sample_len);
            New_data=[New_data split_data];
        end
    else
        for i = 1:div_num
            DivN=floor((m-sample_len)/div_num);
            split_data=data((1+(i-1)*DivN):(sample_len+(i-1)*DivN),1);
            New_data=[New_data split_data];
        end
    end

end
