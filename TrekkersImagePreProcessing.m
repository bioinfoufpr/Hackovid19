function [output matsweep] = TrekkersImagePreProcessing(data)
% Provide the location of the image files and receive the SWEEP projection
% of data and the projection matrix
% process: the image matrix is resized to a 250x250 binary image and vectorized 
% (one image per line). Then the data is projected by Sweep

% This function is based on the SWeeP algorithm developed by the 
% artificial intelligence laboratory applied to UFPR bioinformatics. 
% Please cite the paper if you are going to use this script.
%
% De Pierri (2020) SWeeP: representing large biological sequences datasets
% in compact vectors. Scientific Reports.
%
% link of SWeeP
% https://sourceforge.net/projects/spacedwordsprojection/
% link of Paper
% https://www.nature.com/articles/s41598-019-55627-4
%
% -----------------------------------------------------------------------
% author: Camila Pereira Perico and Monique Schreiner
% HackCovid 2020 - Team: Trekkers
% 
% The team:
%   Monique Schreiner
%   Camila Pereira Perico
%   André Caliari
%   Selma dos Santos Carolino de Andrade
%   Leonardo Custodio
%   Guilherme Taborda Ribas 

 

a = dir(endr);   
lista = {a.name};
data = [];
  for k=3:length(lista)
      vetbin = {}; 
      rgbImage = imread([cell2mat(mat2celllines(endr)) cell2mat(lista(k))]);
      J = imresize(rgbImage,[250 250]);
%       imshow(J)
      vetbin = reshape(dec2bin(J, 8).' - '0', [], 1);
      data = [data; mat2vet(vetbin)];
%       save([nome2 'c' num2str(k)],'vetbin')
      clear vetbin
      disp(k)
  end

  
% start SWeeP projection
compr = size(data,2);
matsweep = orthbase(compr, 600);
output = data*matsweep;
    
end




