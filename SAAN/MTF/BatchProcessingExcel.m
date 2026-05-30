excel_path= '.\Bearing1_3\';    
path_list = dir(strcat(excel_path,'*.xlsx'));          

list_num = length(path_list);
DataX=[ ];
for i=1:list_num
   num= xlsread([excel_path,path_list(i).name]);
   A=num(1:2000,3);
   DataX=[DataX A];
end

save('DataX','DataX')


