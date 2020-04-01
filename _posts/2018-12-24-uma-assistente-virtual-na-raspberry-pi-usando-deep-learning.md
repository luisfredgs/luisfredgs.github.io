---
layout: post
title:  "Uma assistente virtual na Raspberry PI usando deep learning"
resume: "Quem acompanha o meu blog já deve ter visto que há algum tempo eu desenvolvi uma assistente virtual que usa inteligência artificial. Isto já tem mais ou menos um ano e, desde então, muita coisa mudou e eu aprendi coisas novas que poderiam resultar em novos recursos para o projeto."
date:   2018-12-24 11:50:00
categories: [experimentos, deep-learning]
tags: [experimentos, deep-learning]
permalink: /experimentos/uma-assistente-virtual-na-raspberry-pi-usando-deep-learning
status: 1
---

![](/assets/img/uma-assistente-virtual-na-raspberry-pi-usando-deep-learning.png)

Logo que comecei a estudar sobre machine learning, fiquei inquieto para aplicar o conhecimento adquirido em algum projeto, eu queria mesmo praticar. Então, decidir criar uma assistente virtual e logo a aplicação estava pronta. Isto já tem mais ou menos um ano e, desde então, muita coisa mudou e eu aprendi coisas novas que poderiam resultar em novos recursos para o projeto. Também tive a ideia de transferir os modelos da assistente virtual para uma Raspberry Pi. Tive algumas dificuldades para atingir a meta, mas deu tudo certo no final. **[você pode pular para o vídeo no final do post, para conferir o resultado final]**

Eu decidi escrever este post apenas para relatar algumas experiências adquiridas enquanto lidava com o aprendizado de novas metodologias em machine learning, com o objetivo de ampliar as capacidades da minha assistente virtual e também de aprender mais sobre o assunto. Depois de um tempo, eu percebi que poderia ter desenvolvido uma assistente muito melhor, se meu conhecimento em torno do assunto não fosse tão limitado. Pois bem, eu já tinha um projeto voltado para a aplicação prática do meu aprendizado em machine learning e queria me aprofundar.

Percebi que poderia usar redes neurais, mas ainda estava começando e me faltava conhecimento. Eu finalmente consegui alcançar esta meta depois de alguns meses e, no final deste post, você poderá conferir um vídeo mostrando o resultado.

## Voltando no tempo

O vídeo abaixo foi gravado em meados 2017. Meu conhecimento em machine learning era bem limitado naquele momento, mas eu estava muito empolgado com os conceitos novos que havia aprendido.

Com o que eu tinha à disposição naquele momento, só poderia ir pouco além de criar alguns modelos mais simples, como aqueles baseados em SVMs (Support Vector Machines). Embora eu já tivesse algum contato com redes neurais naquela época, só pude aplicá-las parcialmente na POLIANA (e com modelos pré-treinados), em face da pouca familiaridade que tinha com estes algoritmos. Enfim, consegui desenvolver alguns modelos mais básicos para processamento de linguagem natural usando o [scikit-learn](http://scikit-learn.org/) combinado com outras técnicas.

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/l2kQud0Epx8" frameborder="0" allowfullscreen></iframe></center>
</div>

*A voz da assistente foi obtida por meio da API da IBM, a única coisa que não consegui desenvolver por conta própria. Mesmo assim, a POLIANA se saiu razoável naquele primeiro momento.*

Enquanto isso, eu lia cada vez mais sobre redes neurais, e já tinha notado o quão mais precisas elas são. De fato, pude constatar a superioridade destes algoritmos enquanto fazia testes comparativos com **M**áquinas de **V**etores de **S**uporte (SVM — Support vector Machines), que eu já estava usando na POLIANA (por sinal, são excelentes classificadores). Por esta razão, achei que seria oportuno adotá-las por completo. Estava evidente que não seria uma tarefa simples e eu precisava estudar um pouco mais.

## Migrei para redes neurais

E, então, comecei a reunir um material e já tinha uma noção clara do quanto ainda me faltava em termos de conhecimento teórico. Eu tinha os seguintes questionamentos:

* Que tipo de redes neurais eu devo adotar para melhorar a interface de entendimento de linguagem natural da minha assistente virtual?

* Quais ferramentas (libs e frameworks) eu devo utilizar? Quais são mais indicadas para produção?

* Como treinar estas redes? Com quais problemas posso me deparar?

No início, me pareceu intimidadora uma parte razoável da literatura. Isto foi resolvido depois que passei a prestar mais atenção às referências bibliográficas contidas nas últimas páginas dos papers. Assim, algum tempo depois, eu já tinha absorvido bastante coisa e sabia do que iria precisar: **Redes Neurais Recorrentes**.

A POLIANA utiliza reconhecimento de voz para interpretar os comandos enviados a ela. Estes comandos de voz são convertidos para sequências de texto (STT — Speech To Text) e, a partir deste ponto, eu tenho mais liberdade para tratar estes dados de diversas formas. Quando se fala em processamento de sequências, é muito difícil não sugerir o uso de [redes neurais recorrentes]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %}). Isto acontece por conta da sua incrível capacidade para lidar com informações sequenciais, como é o caso dos textos.

Embora sejam mais intensivas computacionalmente, elas costumam ser incrivelmente precisas na identificação de contextos, principalmente quando a ordem das palavras importa. As redes convolucionais também são eficientes para processar textos (e muito mais rápidas de treinar), desde que a ordem das palavras não seja um fator determinante. Redes recorrentes mais básicas são frequentemente limitadas em sua capacidade de identificar informações contextuais em sequências mais longas, por isso as LSTMs ou GRUs são variantes mais indicadas nestes casos. Um outro conceito interessante que eu consegui absorver neste meio tempo foi o [Reconhecimento de Entidades Nomeadas]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %}), que revelou-se bastante útil depois que o apliquei na assistente. No vídeo de demonstração que está mais abaixo neste post é possível entender porque estou utilizando técnicas de reconhecimento e extração de entidades.

## Escolhendo os frameworks de Deep Learning

Quantos às libs e frameworks, eu já tinha contato com o Tensorflow desde as versões iniciais, mas ainda não estava confortável com ele o suficiente. Então, comecei a procurar alternativas. Depois de tentar algumas, cheguei a conclusão de que o Tensorflow era mesmo o melhor para o meu caso, que me dava muito mais flexibilidade. Eu acabei descobrindo um framework bem legal, o [PyTorch](https://pytorch.org/), mas decidi usá-lo em outras situações. O tensorflow é muito verboso, por isso eu decidi adotar o [Keras](https://keras.io/) como uma camada de abstração. Hoje em dia, o keras já está[ oficialmente incorporado](https://www.tensorflow.org/api_docs/python/tf/keras) no código do tensorflow.

Então, eu já possuía o conhecimento e as ferramentas de que precisava para começar, mas algo muito importante me faltava: DADOS.

## Alguns erros básicos que cometi em relação aos dados

Eu realizei alguns experimentos e descobri coisas interessantes sobre o comportamento das redes neurais. Cometi diversos erros, alguns bem básicos, com os quais aprendi muito (os papers que eu havia lido me guiaram durante todo o meu aprendizado, mas o que realmente me ensinou alguma coisa foram os erros que cometi). Lembro de um dia em que passei um domingo quase inteiro treinando apenas um modelo, sem nunca chegar a uma precisão aceitável. Eu alterava diversos parâmetros, como taxa de aprendizado; trocava um otimizador pelo outro; combinava diferentes arquiteturas; alterava o número de camadas ocultas, aumentava o número de epochs, etc. Enfim, para a minha frustração, nada adiantava. Mas, tal como um urubu que fica em cima da carniça, não desisti e demorou mais um tempo até eu descobrir que o problema não estava na arquitetura da rede, mas nos dados que eu estava utilizando. Os dados não apenas eram pouco representativos como também os tinha em menor volume do que o necessário. Aprendi duas lições:

1. Prestar mais atenção aos dados, pois não adianta ter um modelo espetacular se você estiver tentando treiná-lo com dados ruins;

1. Redes neurais não são como outros modelos mais simples, os quais podem ser treinados com dados mais escassos. Elas precisam de ainda mais dados.

## Garimpando dados da internet

Não existem muitos datasets em português — [salvo alguns casos](http://dados.gov.br/) — disponíveis na internet. Também existe o fato de que muitos dados são sensíveis; outros envolvem direitos autorais, o que impede a publicação de algumas destas informações. Nós precisamos produzir mais datasets em português, disponibilizá-los publicamente e diminuir um pouco esta dependência em relação aos gringos. De qualquer modo, eu precisava de dados para treinar as redes neurais da POLIANA, e melhor ainda se fossem em português. Então, eu comecei a “garimpar” estes dados na internet e quando já os possuía em quantidade o suficiente, comecei a “brincar” com eles. Só que eu teria mais uma barreira para transpor.

## Eu não tinha a bendita GPU

GPUs tornam mais rápido o processo de ajuste dos pesos da rede, uma vez que conseguem paralelizar todos os cálculos em cima dos tensores. Eu uso um iMac aqui em casa, mas ele não tem GPU dedicada, o que me gerou transtornos. Ele me serviu bem enquanto criava a versão inicial da POLIANA, só começou a pedir arrego depois que passei a brincar mais frequentemente com redes neurais. A natureza sequencial das redes recorrentes faz com que elas exijam mais capacidade computacional (em comparação com as redes convolucionais, que eu já havia testado com dados de texto, mas sem conseguir obter a performance de predição que eu buscava). Por este motivo, os modelos demoravam muito para treinar com grandes quantidades de dados, muito mesmo. Logo fiquei impaciente, pois precisava de agilidade. Eu tinha três opções, basicamente:

1. Alugar um cloud com GPU para treinar os modelos (AWS ou Azure)

1. Investir em uma máquina nova, mas que não me sugasse todas as economias

1. [Comprar uma eGPU](https://www.macworld.co.uk/feature/mac/best-egpu-mac-3673105/)

## Acabei montando um PC novo

Eu percebi que a segunda opção me sairia menos cara a longo prazo, por isso decidi que seria melhor investir em uma máquina nova e comecei reunir a grana. Na época, a melhor placa de vídeo que minha grana poderia comprar era uma GTX 1060. Eu queria o modelo de 6GB, pelo menos, mas só foi possível pegar o de 3GB. Já era o suficiente para estudar machine learning com redes neurais sem perder muito tempo aguardando a atualização dos pesos da rede. Depois de um bom tempo, eu estava com o tal PC montado e tão feliz quanto mosquito em bunda de cachorro:

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*YTnNuBs_1kWym4rwFRab3A.jpeg)

Devo dizer que foi um bom investimento, considerando que a GPU tem me ajudando bastante.

## Os modelos novos

Consegui desenvolver e testar vários modelos baseados em redes recorrentes [LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %}), GRU e também com redes convolucionais (conforme [Kim, 2014](https://arxiv.org/abs/1408.5882)). Comparei suas performances em diferentes situações e escolhi os melhores (aqueles com acurácia acima de 87% no conjunto de dados de validação) para substituir os antigos modelos estatísticos da POLIANA. Observei que as SVMs conseguem competir bem com as redes neurais em muitas ocasiões. Com os modelos já treinados, eu só precisaria de um local para eles rodarem.

## Eu precisava manter os modelos em um servidor

Inicialmente, a API Rest da assistente rodava num VPS que eu mantinha [na Digital Ocean](https://m.do.co/c/19bd77965778), exclusivamente pra ela. É claro que esta API não ficou disponível lá por muito tempo, afinal são 5,00 USD / mês, um custo que parece bobo, mas que pesa ao longo do tempo. Então, achei melhor buscar uma alternativa.

Eu já tinha visto alguns vídeos no Youtube, bem como artigos de blog, falando sobre a [Raspberry Pi](https://www.raspberrypi.org/). Trata-se de uma famosa plaquinha “mágica” que já vem com Wi-Fi integrado, além de portas USB, HDMI, entrada para cabo de rede e saída de áudio. Além disso, o dispositivo permite a instalação e uso de distribuições Linux. Era tudo o que eu estava precisando.

Como eu não uso esta assistente virtual fora de casa, já que ela destina-se apenas para a realização de experimentos, uma placa Raspberry Pi poderia ser útil na montagem de um servidor caseiro. O processador quad-core de 1.4GHz 64-bit e sua RAM de 1GB não são suficientes para treinar modelos de deep learning, mas é o que basta para rodar estes modelos. Um detalhe interessante sobre esta plaquinha é que, se faltar poder computacional, a Raspberry Pi permite a montagem de clusters para computação distribuída. Assim, dei aquela pesquisada no mercado livre e acabei por adquirir uma placa há alguns dias:

{:.image}
![Raspberry Pi 3 — modelo B+](https://cdn-images-1.medium.com/max/2000/1*x7h7Dy47B0BACyKB1S0X4Q.jpeg)*Raspberry Pi 3 — modelo B+*

Aproveitei para adquirir um cartão de 16GB classe 10, já que a velocidade de I/O era uma preocupação minha (considerei usar um SSD, mas a brincadeira já estava ficando muito cara), e instalei um Linux baseado em Debian. Eliminei a interface gráfica para poupar recursos e transferi a rede neural da assistente para o dispositivo. Algumas horas depois, estava tudo pronto. Com o linux instalado, ficou fácil, pois bastaria configurar o ambiente com todas as ferramentas necessárias para rodar os modelos.

## Finalmente pronto

Foi empolgante ver tudo funcionando na Raspberry Pi. Para além disso, aproveitei todo este embalo para atualizar o aplicativo Android da assistente, incluindo algumas melhorias em relação à versão anterior. No vídeo que está na próxima seção, você poderá notar estas mudanças. O esquema de funcionamento atual está de acordo com a figura abaixo:

{:.image}
![Esquema de funcionamento](https://cdn-images-1.medium.com/max/2048/1*TWoisfddcbURQQh4Jqcv6Q.png)*Esquema de funcionamento*

## Vídeo de demonstração

No vídeo abaixo é possível conferir os resultados das alterações mais recentes que fiz na assistente virtual.

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/fJAYFaeiAvI" frameborder="0" allowfullscreen></iframe></center>
</div>

## Próximos passos

Atualmente a POLIANA executa um conjunto de ações muito específicas, baseadas na inferência que ela consegue obter em relação às intenções do usuário durante cada solicitação. As respostas dela estão restritas a algumas dezenas de possibilidades. Acontece que eu sempre quis mudar isto. Por este motivo, estou fazendo mais algumas alterações e testando o uso de modelos baseados em [Sequence2Sequence]({% post_url 2018-04-10-o-que-e-sequence-to-sequence-em-deep-learning %}). Isto permite que a assistente mantenha uma conversa ainda mais natural, ao ampliar o seu escopo de conversação. Contudo, é possível que leve um tempo até chegar a um nível de maturidade razoável que justifique a adoção definitiva de seq2seq. Enfim, tem muita coisa nova para experimentar.

## Veja também

Parte do conhecimento necessário para criar uma assistente virtual nos moldes do projeto acima pode ser consultada aqui mesmo no blog. Por meio dos links abaixo.

1. [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

2. [Processamento de Linguagem Natural, do zero à produção](https://www.udemy.com/course/processamento-de-linguagem-natural-do-zero-a-producao/?referralCode=58A9E255EE0A65ACF871)

3. [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})

4. [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

5. [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})
