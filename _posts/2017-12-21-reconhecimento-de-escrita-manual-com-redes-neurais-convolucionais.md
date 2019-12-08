---
layout: post
title:  "Reconhecimento de escrita manual com Redes Neurais Convolucionais"
resume: "No final deste post você terá uma visão geral sobre o funcionamento das redes neurais convolucionais e saberá como usá-las para criar um modelo capaz de fazer o reconhecimento de escrita manual, mais particularmente reconhecer dígitos escritos à mão em imagens"
date:   2017-12-21 11:50:00
categories: [computer-vision, deep-learning]
tags: [computer-vision, deep-learning]
permalink: /deep-learning/reconhecimento-de-escrita-manual-com-redes-neurais-convolucionais
---

{:.image}
![](/blog/assets/img/reconhecimento-de-escrita-manual-com-redes-neurais-convolucionais.png)

{:.intro}
*No final deste post (vídeo no final) você terá uma visão geral sobre o funcionamento das redes neurais convolucionais e saberá como usá-las para criar um modelo capaz de fazer o reconhecimento de escrita manual, mais particularmente reconhecer dígitos escritos à mão em imagens.*

O reconhecimento de escrita manual — também conhecido como *handwriting recognition*, do inglês — é uma aplicação de software que ainda encontra bastante demanda hoje em dia. Se você já usou o [GoodNotes](http://www.goodnotesapp.com/user-guide/handwriting-recognition.html) no iPad, então sabe do que eu estou falando. O reconhecimento de assinaturas escritas à mão a partir de documentos digitalizados/escaneados, a captura de endereços postais escritos em envelopes, ou informações monetárias escritas em cheques bancários estão entre as aplicações mais comuns desta técnica.

De um modo geral, há duas situações nas quais isto pode ser feito:

**On-line:** Conforme o texto estiver sendo escrito, ocorrerá a conversão do sinal obtido a partir do traçado para códigos que correspondem às letras, os quais poderão ser utilizados no computador e em aplicativos de processamento de texto. Isto corre, por exemplo, no aplicativo [GoodNotes](http://www.goodnotesapp.com/user-guide/handwriting-recognition.html).

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*IS2pA0hWLdH4fi_7.png)

**Off-line:** O cerne deste tipo de tarefa está em transcrever para dados eletrônicos as informações escritas à mão em qualquer folha de papel que tenha sido digitalizada, ou mesmo textos presentes em imagens naturais.

{:.image}
![](https://cdn-images-1.medium.com/max/2048/0*TTb25AF_JbJ70SnZ.jpg)

Sem dúvidas, este último cenário é talvez o que mais pode se beneficiar desta técnica, pois há muitos documentos impressos com dados escritos à mão cujas informações podem possuir grande valor, inclusive estratégico. Meu objetivo neste post é abordar justamente a aplicação do reconhecimento de escrita manual neste cenário, fazendo uso de um modelo gerado a partir de uma rede neural convolucional, com base em um conjunto de dados de treino, para reconhecer caracteres numéricos presentes em imagens ainda não vistas pelo modelo. Nós utilizaremos o [dataset MNIST](http://yann.lecun.com/exdb/mnist/), que já é bem conhecido, e o framework [Keras](https://keras.io/). O dataset MNIST possui 60k exemplos de treino e 10k exemplos de teste, sendo que todos estes dados correspondem a dígitos no intervalo 0–9, escritos à mão e digitalizados. Cada dígito é representado por uma imagem monocromática com 28x28 pixels.

Uma vez que nosso tutorial se baseia no uso de redes neurais convolucionais, é útil ter um certo conhecimento sobre como funciona esta arquitetura. Por este motivo, irei resumir um pouco deste funcionamento, mas é importante que você tenha uma visão mais completa em torno deste assunto. Por isso, eu recomendo que você [leia este post](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/).

Leia também — [Machine Learning e a matemática da aprendizagem supervisionada]({% post_url 2017-11-27-machine-learning-a-matematica-da-aprendizagem-supervisionada %})

## Uma visão geral das Redes neurais convolucionais

Não há muita diferença entre um rede neural regular e uma convolucional (doravante denominada ConvNet). O que difere estas arquiteturas é que as ConvNets basicamente foram concebidas para extrair features a partir de dados brutos presentes nos pixels de uma imagem. Elas possuem camadas com neurônios arranjados em três dimensões: largura, altura e profundidade. Normalmente, utiliza-se um stack de três tipos de camadas principais (lembre-se de que isto não é uma regra):

**Convolutional Layer :** (Camada Convolucional) — Para cada sub-região presente na imagem, esta camada aplica um conjunto de operações matemáticas (transformações lineares) para produzir um simples valor no mapa de features que é gerado como saída da camada (cada convolução pode gerar um ou mais mapas de features). Uma [função de ativação do tipo ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (Rectified Linear Unit) é comumente aplicada com o objetivo de promover a não linearidade ao modelo, uma das funções mais populares para esta finalidade. A camada convolucional pode gerar um ou mais mapas de features como saída.

{:.image}
![Uma animação que ilustra como uma convolução acontece](https://cdn-images-1.medium.com/max/2000/0*UWaRHLnfmmzKOUMg.gif)*Uma animação que ilustra como uma convolução acontece*

**Pooling Layer :** (Camada de subamostragem): A camada Pooling computa o máximo de valores obtidos em cada filtro da camada convolucional, só que o resultado desta operação é apenas uma pequena amostra do que a camada recebeu como entrada (uma pequena amostra com as informações mais representativas obtidas com cada filtro na camada convolucional). Uma vantagem desta camada é que ela reduz a dimensionalidade ao mesmo tempo em que consegue manter as informações mais úteis. É um dos motivos por trás da eficiência computacional das ConvNets

**Fully-Connected Layer :** (Camada densamente conectada) é a camada final de uma ConvNet, onde se coloca o classificador. Trata-se de uma camada densamente conectada, na qual cada neurônio conectado a todos os outros de uma camada anterior e posterior. Basicamente, a FCL recebe uma entrada (que pode ser a saída de uma camada convolucional, ReLU ou Pooling) e gera como saída um vetor de dimensionalidade N, onde N é o número de classes que o modelo precisa predizer.

{:.image}
![](https://cdn-images-1.medium.com/max/2048/0*QilofsgLEudcoSyA.png)

A camada final de uma ConvNet terá reduzido todos os dados brutos da imagem a um simples vetor com os scores atribuídos às classes. Assim, esta arquitetura transforma a imagem original, camada por camada a partir dos valores dos pixels originais em scores de classes (em nosso caso, scores que representam os dígitos 0–9).

## Exemplo de aplicação: Reconhecendo escrita manual

Será preciso que as seguintes bibliotecas estejam instaladas na sua máquina. Mas você nem precisa se preocupar com isto caso use o [Google Colab](https://colab.research.google.com), que já vem com tudo isso instalado (e mais um monte de outras bibliotecas essenciais) e ainda tem GPU gratuita, com um máximo de 12 horas de uso à sua disposição:

* **Anaconda** — O Anaconda é o ambiente ideal para quem trabalha com ciência de dados e Machine Learning em Python. Baixe o ambiente clicando [aqui](https://www.anaconda.com/download/).

* **OpenCV** — Library voltada para o desenvolvimento de aplicações em visão computacional. Neste caso, você precisa ter instalada a implementação em Python. Se tiver o Anaconda instalado, basta rodar este comando: $ conda install -c conda-forge opencv ou $ conda install -c conda-forge opencv=3.2.0

* **Keras** — Uma library minimalista que usa a linguagem Python para nos permitir criar uma grande variedade de modelos de deep learning em cima dos frameworks [TensorFlow](https://www.tensorflow.org/) e [Theano](http://www.deeplearning.net/software/theano/), abstraindo a maior parte da complexidade envolvida no processo de criar modelos puramente nestes dois frameworks. Para instalar, siga as instruções presentes no próprio [site da ferramenta](https://keras.io/).

Além disso, irei supor que você já saiba programar em Python e que conheça ao menos o básico do Keras.

## Montando a rede neural

Você pode [baixar o código](https://github.com/luisfredgs/keras-cnn-handwriting-mnist) **atualizado**, que foi criado como um notebook na ferramenta Jupyter (você pode baixar o Jupyter diretamente pelo ambiente Anaconda). O código foi desenvolvido durante o vídeo abaixo, mas foi atualizado algum tempo depois, com algumas falhas corrigidas. Uma outra coisa que você pode fazer é copiar este código para um notebook na [Kaggle](https://www.kaagle.com) ou no [Google Colab](https://colab.research.google.com/).

O vídeo, que começa com uma breve introdução às redes convolucionais, abordando as classes do Keras que são necessárias para criar a rede, mostra em detalhes como montar uma ConvNet para reconhecer dígitos escritos à mão

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/FhwzOaEMk6Y" frameborder="0" allowfullscreen></iframe></center>
</div>

## Referências

* [Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)

* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)

* [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

* [A guide to convolution arithmetic for deep learning — Vincent Dumoulin and Francesco Visin](https://arxiv.org/pdf/1603.07285.pdf)

* [Deep Learning — *Ian Goodfellow* and Yoshua Bengio and Aaron Courville](http://amzn.to/2DqqiVA)
