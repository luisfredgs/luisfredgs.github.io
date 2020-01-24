---
layout: post
title:  "Dicas para aprender Machine Learning"
resume: "O que é preciso para começar a aprender Machine Learning? Eu diria que o principal é que você seja curioso, em primeiro lugar. Depois, alguma afinidade com matemática e programação."
date:   2018-01-18 11:50:00
category: machine-learning
tags: [machine-learning]
permalink: /machine-learning/dicas-para-aprender-machine-learning
---

![](/assets/img/linear_algebra.png)

O que é preciso para começar a aprender Machine Learning? Eu diria que o principal é que você seja curioso, em primeiro lugar. Depois, alguma afinidade com matemática e programação. contudo, ao contrário do que provavelmente já te fizeram crer, você não precisa ser nenhum PhD para aprender machine learning, muito embora isso fosse ajudar bastante na sua jornada. No mais, leia o restante deste post para descobrir o quão perto de aprender machine learning você pode está *[**há um vídeo no final**]*.

No final deste texto, você saberá sobre:

1. Quais são os pré-requisitos para aprender Machine Learning;

1. Quais livros poderá usar para aprender sobre o assunto;

1. Qual linguagem de programação preciso aprender?

1. Quais ferramentas computacionais são mais usadas?

1. Preciso mesmo aprender tudo isso?

Antes de prosseguir, aproveita que já veio aqui e participa do grupo que criei no Telegram, para promover discussões sobre o tema aprendizagem de máquina e tirar dúvidas: [https://t.me/joinchat/Omx7D1hGjM2K_8YfwwntYw](https://t.me/joinchat/Omx7D1hGjM2K_8YfwwntYw)

De um modo geral, em machine learning nós precisamos desenvolver um modelo matemático que possa representar certas suposições sobre um problema do mundo real que estamos tentando resolver e confirmar nossas suposições, ou hipóteses, sobre tal problema. Assim, nós precisamos elaborar uma **função matemática** que nos permita medir **o quão bem estas suposições correspondem à realidade**. Para tanto, nós precisamos de um **algoritmo** que seja exposto a um **conjuntos de dados representativos** em relação ao problema (os dados de treino relativos ao domínio da tarefa), e que seja capaz de reconhecer padrões nesses dados, promover um ajuste iterativo da tal função matemática em relação aos parâmetros contidos nos dados, **minimizando uma função custo** (também conhecida como função de perda), a qual indica o qual longe ou perto estamos de uma função ótima, pois este é o objetivo. Durante esse processo, **a matemática é utilizada diversas vezes** por meio de uma **camada de software**, no que pode chegar a ser uma mistura de álgebra linear, cálculo diferencial, teoria da probabilidade e outras aplicações matemáticas. Eu sei que esta introdução parece meio confusa, mas a esta altura, você já percebeu do que precisará para trabalhar com Machine Learning, certo? Pois é, o ML é, fundamentalmente, **matemática aplicada e software**. Uma vez que você tenha aprendido os conceitos de que precisa, esta introdução lhe parecerá muito menos confusa.

## Porque a álgebra linear é um pré-requisito tão importante para o Machine Learning?

A álgebra linear é uma área da matemática que é largamente utilizada em diversos ramos da engenharia e da ciência. Uma boa compreensão da álgebra linear é essencial se você pretende trabalhar com algoritmos de ML. Não é meu objetivo aprofundar o assunto de álgebra linear neste post, mas se eu puder resumir em poucas palavras no que você deveria se concentrar, a princípio eu diria (a ordem faz diferença): **Escalares**, **Vetores**, **Matrizes** e **Tensores**. Além disso, é importantíssimo que você compreenda os seguintes tópicos:

* Multiplicação entre matrizes e vetores

* Matriz identidade e matriz inversa

* Dependência linear

* Transformações lineares

* Autovalores e Autovetores

Há outros tópicos, é claro, mas estes são os mais importantes. Em machine learning, fazemos um uso intensivo de operações envolvendo tais estruturas, sem as quais seria muito difícil trabalhar.

## Cálculo diferencial

A diferenciação é uma das mais importantes ferramentas na ciência, como na física, por exemplo, quando estamos tentando modelar a dinâmica de um determinado sistema. Basicamente, as diferenciações relacionam a taxa de variação de uma quantidade específica à outras propriedades presentes no sistema. Para citar um exemplo, a dinâmica de um modelo neural em deep learning utiliza diferenciação (derivação parcial) para atualizar os pesos da rede de acordo com alguma regra específica. Um exemplo mais concreto dentro do mesmo contexto seria o algoritmo de retropropagação do erro (backpropagation) que, por meio do *gradiente descendente* da função de perda, utiliza derivadas parciais para atualizar os pesos da rede com base no sinal de erro produzido pelos neurônios na camada de saída.

> *“ a dinâmica de um modelo neural em deep learning utiliza diferenciação (derivação parcial) para atualizar os pesos da rede a partir do gradiente descendente da função de perda”*

É interessante, portanto, que você compreenda estes tópicos:

* Derivadas, bem como as derivadas parciais e regra da cadéia

* Derivação com múltiplas variáveis

* Integrais (normalmente apenas as integrais de primeira ordem)

* Cálculo de vetor e função gradiente

Não! Você não vai precisar calcular integrais e derivadas de uma maneira direta para que seus modelos funcionem. Tudo isso é abstraído por pacotes de ferramentas que foram desenvolvidas especialmente para o caso. Tudo o que você tem que fazer é usar algumas linhas de código corretamente. A mensagem que precisa ser entendida aqui é: Você precisa ao menos saber o que está acontecendo enquanto seu modelo de machine learning está sendo treinado. Esse entendimento influencia na escolha dos hiperparâmetros que vão alterar a dinâmica da aprendizagem do seu modelo, por exemplo. Isso influencia diretamente na qualidade dos modelos. Então, conhecimento, nessas horas, é poder!

## Teoria da probabilidade

A incerteza é um conceito chave em inteligência artificial e reconhecimento de padrões. É neste ponto que surge a necessidade de uma ferramenta para quantificar e representar a incerteza. Por tal motivo, a probabilidade é uma ferramenta de grande interesse na área de  inteligência artificial. Em ML você vai lidar basicamente com a aleatoriedade e, portanto, um entendimento básico em teoria da probabilidade (principalmente a interpretação Bayesiana) é desejável para que você possa trabalhar com os algoritmos e gerar modelos preditivos.
> *“Em ML você vai lidar basicamente com a aleatoriedade e, portanto, um entendimento básico em estatística e probabilidade (principalmente a interpretação Bayesiana) é desejável.”*

Um exemplo clássico de aplicação da probabilidade em machine learning é o filtro de spam. Qual a probabilidade de um determinado e-mail ser um SPAM? Qual a probabilidade de um cliente ser um bom pagador ou um inadimplente? Frequentemente, a saída do seu modelo será uma distribuição de probabilidades que corresponda às categorias que você está tratando na variável resposta.

## Estatística inferencial

Muita gente não fala, muitos cursos de machine learning que você encontra por aí não mencionam isso. Mas, um passo crucial para a aceitação e implantação de um modelo de machine learning em produção é a validação dele. Você precisará planejar com cuidado uma série de experimentos com o seu modelo, testá-lo em diferentes situações e validá-lo estatisticamente. O modelo X é melhor do que o modelo Y? O quanto ele é melhor? Para responder a essas questões, você terá que se apoiar em importantes ferramentas da estatística inferencial, tais como os **testes de hipótese**. Qualquer livro de estatística vai te ensinar como conduzir tais testes, fora inúmeros vídeos que se encontram no Youtube.

## Quais linguagens de programação são mais usadas em Machine Learning

Você vai precisar de uma ferramenta computacional para poder obter e testar seus modelos em machine learning, afinal, fazer isto manualmente é impraticável. Você precisa entender os conceitos, o que é importantíssimo, mas não significa que vai implementar todos os cálculos na unha. Você vai precisar de alguma linguagem de programação e frameworks de machine learning.

Se você ainda não programa em nenhuma linguagem, ou pelo menos em nenhuma das que vou listar aqui, então tente começar por aquelas mais valorizadas no mercado.

As linguagens de programação mais utilizadas pela comunidade e mais exigidas pelas empresas são:

* **Python**

* **R**

* C++

* Matlab

* Java

* Scala

*“ Se você pretende atuar no mercado como profissional destas áreas, então tente buscar conhecimento principalmente em Python ou R, se possível nas duas linguagens”*

Hoje em dia, **Python** e **R** são as mais utilizadas, seja por pesquisadores ou por empresas na área de Data Science e IA. Quase não se vê anúncios de vagas exigindo conhecimentos em Matlab, C++, Java ou Scala para atuar na área. Se você pretende atuar no mercado como profissional de Data Science ou Machine Learning, então tente buscar conhecimento principalmente em Python ou R, se possível nas duas linguagens. Uma grande maioria das *toolboxes* disponíveis para machine learning — ao menos as que são baseadas em linguagens open source — que existem hoje estão implementadas nestas duas linguagens; a maior parte dos cursos online ou dos livros voltados para este tema se baseiam nestas duas ferramentas. Então, é fundamental que você tente se aprofundar no uso destas linguagens de programação.

Além disso, existem alguns frameworks e libraries que são muito requisitados nas vagas que costumo encontrar no Linkedin, por exemplo. No ecossistema da linguagem Python, posso citar alguns:

* Tensorflow
* Scikit-learn
* Pandas
* Numpy
* Scipy

São ferramentas já bastante consolidadas na área e muito usadas em prototipação e produção. Vale a pena investir tempo em entender seu funcionamento.

### Quais livros ou cursos online você me recomenda para estudar machine learning?

Esta é uma pergunta que eu recebo com muita frequência. Por isso, decidi gravar o vídeo abaixo, onde eu indico alguns dos livros que utilizo para estudar e também alguns cursos online que, apesar de eu ainda não ter feito nenhum deles, sei que são cursos muito bem conceituados e recomendados pela comunidade. No vídeo eu também apresento um resumo dos pré-requisitos que mencionei nos parágrafos anteriores, bem como algumas dicas sobre onde buscar as informações mais atualizadas sobre o tema.

{:.note}
***ATENÇÃO**: Aos 07:54, eu na verdade estou querendo me referir ao **k-Nearest Neighbors** (o K-NN). Há uma grande diferença em relação ao K-Means, o qual tenta fornecer um agrupamento de dados, com separações bem definidas, baseado na similaridade entre estes dados. Falo mais sobre o K-means [nesse post](https://luisfredgs.github.io/machine-learning/clustering-analysis-an-introduction-to-unsupervised-learning). Peço desculpas pelo equívoco, pois só notei depois de rever o vídeo, logo após ser publicado no canal. Todos os links aos quais me refiro durante a apresentação dos slides estão presentes na descrição do vídeo.*

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/9aCUXJXPHGw" frameborder="0" allowfullscreen></iframe></center>
</div>

### E se eu não quiser aprender tudo isso para poder usar machine Learning?

As APIs cognitivas da Google (Google Cloud Machine Learning), Microsoft (Azure Machine Learning) e IBM (IBM Watson) existem para facilitar a vida de quem não quer ter o trabalho de estudar tudo o que foi dito nos parágrafos anteriores para poder criar sistemas de inteligência artificial. De fato, se você tiver o conhecimento em programação necessário para utilizar os *endpoits* destas APIs, poderá criar sistemas inteligentes usando modelos prontos e sem se preocupar com teoria. Apenas tenha em mente que este uso tem um custo envolvido, pois normalmente estes serviços cobram um valor específico a partir de um determinado volume de requisições. Entretanto, aprender a teoria continua sendo importante se você quiser extrair o máximo do potencial destas APIs e modelar sistemas mais robustos.

Não deixe de comentar aí na parte de baixo desta página, caso você tenha ficado com qualquer dúvida ou se tem alguma sugestão para novos posts ou vídeos no canal.

## Leia também

* [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

* [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})

* [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

* [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})
