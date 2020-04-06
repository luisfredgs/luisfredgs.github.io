---
layout: post
title:  "Como o Youtube usa o deep learning para saber qual é o próximo vídeo que você quer assistir"
resume: "Já percebeu que quando você acessa a página inicial do Youtube, os vídeos que aparecem convenientemente logo no topo têm a mais alta probabilidade de serem os vídeos que você gostaria de ver? Você já se perguntou como a maior plataforma de vídeos do mundo sabe qual é o próximo vídeo que você quer assistir? Ao final deste post, você terá uma ideia de como o Youtube utiliza o deep learning para chegar a este resultado."
date:   2017-11-27 11:50:00
categories: [deep-learning]
tags: [deep-learning]
permalink: /deep-learning/como-o-youtube-usa-o-deep-learning-para-saber-qual-e-o-proximo-video-que-voce-quer-assistir
status: 0
---
{:.image}
![](/assets/img/como-o-youtube-usa-o-deep-learning-para-saber-qual-e-o-proximo-video-que-voce-quer-assistir.png)

{:.intro}
*Já percebeu que quando você acessa a página inicial do Youtube, os vídeos que aparecem convenientemente logo no topo têm a mais alta probabilidade de serem os vídeos que você gostaria de ver? Você já se perguntou como a maior plataforma de vídeos do mundo sabe qual é o próximo vídeo que você quer assistir? Ao final deste post, você terá uma ideia de como o Youtube utiliza o deep learning para chegar a este resultado.*

A Google recentemente deu um grande passo em direção à próxima geração de seus aplicativos baseados em inteligência artificial, introduzindo tecnologia neural que vem garantindo performances no [estado da arte](https://pt.wikipedia.org/wiki/Estado_da_arte). Um grande exemplo disto é a segunda geração do Google Tradutor, no qual os modelos estatísticos de tradução foram substituídos por uma nova abordagem chamada **Neural Machine Translation** (Tradução Neural de Máquina), conforme você poderá conferir [neste post](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/), no blog da Google.

Um outro produto Google que passou por uma mudança similar recentemente foi o Youtube, no qual passaram a adotar técnicas de *deep learning* com o objetivo de predizer qual é o próximo vídeo que nós queremos assistir na plataforma, constituindo a base para o seu novo sistema de recomendação. Os resultados da pesquisa foram descritos e compartilhados [neste paper](https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/45530.pdf).

A versão anterior deste sistema de recomendação contava com uma abordagem baseada em **fatoração de matrizes**, em conjunto com a **filtragem colaborativa**, a qual se baseia, por sua vez, no feedback dado pelos usuários (likes, views, etc). Um *paper* interessante sobre este assunto pode ser [encontrado aqui](https://arxiv.org/ftp/arxiv/papers/1503/1503.07475.pdf). O novo sistema de recomendação é composto por duas redes neurais, cada uma responsável por uma tarefa específica: ***geração de candidatos*** e ***rankeamento**, *respectivamente. A figura abaixo mostra uma visão geral do funcionamento deste sistema.

{:.image}
![](https://cdn-images-1.medium.com/max/2048/0*xQJC_w0RS_AF78M6.png)

### Geração de candidatos

Esta rede neural gera uma lista dos vídeos que têm a maior probabilidade de serem relevantes para quem estiver acessando a plataforma a qualquer momento. Utiliza como entrada, informações baseadas no histórico de atividades dos usuários. Nesta etapa, a filtragem colaborativa torna-se um elemento importante, pois se baseia em determinadas ações dos usuários que constituem-se num forte indicativo de que um determinado vídeo seja relevante. Se, por exemplo, um vídeo específico foi assistido na íntegra por várias pessoas, isto pode ser um grande sinal de que este vídeo seja relevante dentro do contexto no qual esteja inserido. A quantidade de *likes* que este vídeo recebeu também é um importante feedback a ser utilizado como entrada. A similaridade entre os usuários que assistem ao vídeo se baseia em determinadas características compartilhadas entre eles, tais como os dados demográficos, textos utilizados no campo de busca, vídeos que assistiram, de que maneira eles descobrem novos vídeos, etc. Esta similaridade também “entra no jogo”.

### Rankeamento

Para que o sistema de recomendação consiga ser realmente efetivo, é preciso que seja garantido um alto valor para o recall (*a fração dos vídeos que são realmente relevantes em relação ao número total de registros candidatos gerados*). A rede neural de rankeamento possibilita isto ao atribuir um score para cada vídeo, com base numa *função objetivo* que utiliza um maciço conjunto informações capazes de descrever tanto os vídeos quanto os usuários que os assistem. Apenas os vídeos com scores mais altos serão exibidos na lista como recomendação para os usuários, rankeados com base em seu score.

### Arquitetura do modelo

Essencialmente, o novo sistema de recomendação do Youtube é um problema de classificação em múltiplas classes (na ordem de milhões), na qual o modelo recebe bilhões de parâmetros como entrada. A figura abaixo ilustra a arquitetura da rede neural utilizada para selecionar vídeos que tenham maiores probabilidades se serem relevantes para os usuários (geração de registros candidatos), dentro do contexto específico de cada um.

{:.image}
![](https://cdn-images-1.medium.com/max/2612/0*HvOGyZS-gHoYQ0c2.png)

É possível perceber que a rede, cujo fluxo se dá de baixo para cima, recebe como entrada algumas features incorporadas, tais como a média dos vídeos visualizados, dos *tokens* que formam os textos digitados no campo de busca, áreas geográficas dos usuários, o gênero deles, a quanto tempo o vídeo está publicado na plataforma, etc. A média destes dados incorporados em representação vetorial é calculada imediatamente antes de eles serem concatenados, com o objetivo de fixar o tamanho dos vetores, de modo a adequá-los à entrada nas camadas ocultas da rede profunda. Todas as camadas ocultas são totalmente conectadas (cada nó em uma camada está conectado a todos os nós da camada anterior e também da camada seguinte). Durante a fase de treino, a função de perda é minimizada por meio da técnica do gradiente descendente na saída da função *softmax *(responsável por representar a probabilidade de classe). Em seguida, é feita uma procura pelo registro vizinho mais próximo de modo a determinar os ***N*** vídeos mais relevantes a serem exibidos na lista.

Considerando que os usuários normalmente preferem conteúdo mais novo e atualizado, o sistema precisa levar em conta, ainda, as milhares de horas em vídeo que são enviadas para a plataforma a cada minuto. Ou seja, os novos vídeos também precisam “entrar na lista” para serem recomendados. Os dados de treino utilizados para “ensinar” a rede são gerados a partir de TODOS os vídeos visualizados na plataforma, até mesmo aqueles que estão incorporados em páginas de sites externos.

Uma arquitetura similar àquela utilizada na fase de geração de candidatos entra em ação logo em seguida, para rankear os registros candidatos obtidos por meio da rede ilustrada na figura anterior. A diferença neste ponto é que os pesquisadores utilizaram *regressão logística* para atribuir um score independente à cada vídeo a ser exibido na lista de recomendados. A figura abaixo ilustra este procedimento:

{:.image}
![](https://cdn-images-1.medium.com/max/2128/0*lScbklnqqXgm8NtA.png)

Centenas de features (idioma do vídeo, exibições anteriores na lista de recomendação, quanto tempo levou desde que cada vídeo candidato foi visto pela última vez, quantos vídeos do canal em questão foram vistos pelo usuário, quando foi a última vez que o usuário assistiu a um vídeo do tópico em questão, etc) alimentam a rede, que possui todas as suas camadas completamente conectadas, tal como ocorre na rede neural para geração de candidatos. Nesta etapa, a lista de vídeos recomendados é ordenada de acordo com o score atribuído a cada vídeo e, então, exibida na tela do usuário.

Em ambos os modelos, as features são conectadas em uma primeira camada mais larga, ao passo em que as camadas seguintes são completamente conectadas e compostas de Unidades Lineares Retificadas ([ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))) como funções de ativação.

Parte do processo de criação do modelo se deu de modo offline. No entanto, para determinar a efetividade do modelo, os pesquisadores fizeram experimentos em produção com base em testes A/B, pois assim conseguiriam medir mudanças sutis, tais como: taxas de cliques, tempo de visualização e muitas outras métricas que medem o engajamento dos usuários.

O sistema de recomendações do Youtube é responsável por ajudar mais de um bilhão de usuários a encontrar conteúdo personalizado diariamente, nesta que é a maior plataforma de vídeos do mundo. Para o sistema em si, trata-se de uma tarefa que envolve um desafio imenso, considerando a necessidade de obter uma boa performance numa escala tão grande e num intervalo de tempo que fica na ordem dos milissegundos. Da próxima vez em que você acessar a página inicial do Youtube e ver todos aqueles vídeos que combinam exatamente com o que você mais gosta de assistir, terá uma boa ideia do que é preciso para que esta “mágica” aconteça.


* [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

* [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})

* [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

* [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})