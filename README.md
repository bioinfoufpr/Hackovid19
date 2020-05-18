## QUEM SOMOS
Somos a equipe Trekkers formada pelos alunos de mestrado em Bioinformática da UFPR no Laboratório de Inteligência Artificial. Estamos participando do HackCovid 2020 e nosso desafio foi o #d051 IA e Ciência de Dados para Apoio à Decisão Clínica.

Mais detalhes de nosso projeto pode ser encontrado em nossa página no DevPost: https://devpost.com/software/teste-jnk5z4

### Integrantes:
André L. Caliari Costa
Camila Pereira Perico
Guilherme Taborda Ribas
Leonardo Custódio
Monique Schreiner
Selma dos Santos C. de Andrade

### CONTATO
Site dos discentes: https://www.bioinfodiscentes.com.br/
Site oficial do programa de pós-graduação: http://www.bioinfo.ufpr.br/
email: bioinfodiscentes@gmail.com



## PROJETO
Com o uso de ferramentas de inteligência artificial, foi desenvolvido a ferramenta Trekkers, que permite classificar imagens de raio X de pulmão em três categorias:
- normal ou saudável
- sinais de estar acometido por doença pulmonar
- indicativo de COVID-19

Essa ferramenta é voltada para o profissional da saúde que necessita realizar um diagnóstico rápido para evitar que casos de COVID-19 se tornem ainda mais disseminados. E ainda auxilia na identificação de possíveis patologias pulmonares. Fornecemos ainda o indicativo com um percentual de confiabilidade.

Faça o upload da imagem de raio X de peito no botão UPLOAD, e aguarde alguns instantes para o processamento.
Você receberá o indicativo com um percentual de confiabilidade, que auxiliará em seu diagnóstico. 


## A FERRAMENTA

Esta ferramenta ainda está em fase de desenvolvimento. Ela foi desenvolvida em MATLAB.
fornecemos aqui o script de pré-processamento das imagens, utilizando o SWeeP, preparando os dados para o treinamento de uma rede MLP, como utilizada em nosso projeto.
Para utilizar este script, primeiramente você precisa realizar o download do SWeeP, que possui as funções necessárias.
[SWeeP Ferramenta](https://sourceforge.net/projects/spacedwordsprojection/)

Para usar essa ferramenta, por gentilza, cite o [artigo](https://www.nature.com/articles/s41598-019-55627-4).

De Pierri, C.R., Voyceik, R., Santos de Mattos, L.G.C. et al. SWeeP: representing large biological sequences datasets in compact vectors. Sci Rep 10, 91 (2020). https://doi.org/10.1038/s41598-019-55627-4

