---
layout: post
title:  "Classificando textos com Machine Learning"
resume: "A classificação de textos é uma das mais importantes aplicações do processamento de linguagem natural hoje em dia. É o que possibilita diversas interações entre usuários e uma infinidade de ferramentas computacionais disponíveis hoje. Neste post, você verá como isto funciona direto no código Python!"
date:   2018-05-07 11:50:00
categories: [nlp, machine-learning]
tags: [nlp, machine-learning]
permalink: /machine-learning/classificando-textos-com-machine-learning
---

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/uJpU0rYUokM" frameborder="0" allowfullscreen></iframe></center>
</div>

A classificação de textos é uma das mais importantes aplicações do processamento de linguagem natural hoje em dia. É o que possibilita diversas interações entre usuários uma infinidade de ferramentas computacionais disponíveis hoje. Entre estas ferramentas, estão os *chatbots*. Neste post, você verá como classificar textos com machine learning e a linguagem Python. Você vai precisar do framework [Scikit-Learn](http://scikit-learn.org/stable/) instalado na sua máquina. Eu sugiro que instale a ferramenta [Anaconda](https://anaconda.org/), com a qual é muito fácil instalar diversas outras ferramentas do ecossistema python, incluindo o Scikit-Learn. Eu vou assumir que você já conheça a linguagem Python e já possui um entendimento básico do framework acima mencionado. Este post acompanha um vídeo, cujo link está no final. Você também poderá baixar o [código no github](https://github.com/luisfredgs/machine-learning-text-classification).

## O dataset

O “20 Newsgroups” é um conjunto de dados disponível publicamente para uso em pesquisas. Contém aproximadamente 20k documentos, divididos em cerca de 20 categorias. É um data set popular e conhecido pelo seu uso tarefas de classificação de textos em machine learning. Os conteúdos do data set estão separados por categoria e algumas destas categorias são muito próximas: ***comp.sys.ibm.pc.hardware*** e ***comp.sys.mac.hardware***. Há outras categorias que não possuem quaisquer semelhanças: ***rec.autos*** e ***talk.politics.misc***. Nosso objetivo aqui é obter um modelo capaz de classificar textos em algumas destas categorias.

{:.image}
![O dataset 20 Newsgroups](https://cdn-images-1.medium.com/max/2000/0*ry5TWGyyiwFDw7O9.png)*O dataset 20 Newsgroups*

### Importando nossas classes, o dataset e demais funções

Nós contaremos com dois algoritmos diferentes para gerar o nosso modelo: Um **SVM (por meio do estimador SGDClassifier)** e uma rede neural Perceptron Multicamadas, que nada mais é do que **uma rede neural feed-forward**. O objetivo é experimentar um estimador de cada vez, para averiguar a performance de cada um no conjunto de dados validação. Em seguida, usaremos nosso modelo para fazer predições em dados que não foram vistos no conjunto de treino. No vídeo que acompanha este post, eu experimentei os dois estimadores e foi possível notar que a rede neural se saiu melhor. Neste post, entretanto, você verá que eu deixei comentada a linha que permite treinar o modelo via SGDClassifier, usando apenas o MPLClassifier. Mas basta que você remova o comentário no código, para que possa testar com ambos os estimadores, um de cada vez, como fiz durante o vídeo.

{% highlight python %}
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# um classificador linear que utiliza o Gradiente Descendente Estocástico como método de treino. 
# Por padrão, utiliza o estimador SVM.
from sklearn.linear_model import SGDClassifier
# Uma rede neural Perceptron Multicamadas
from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
{% endhighlight %}

## Limitando a quantidade de categorias e obtendo os dados de treino

Apenas por uma questão de agilidade, iremos reduzir a quantidade de dados a serem processados. Conseguiremos isto ao limitar o número de categorias em apenas duas. Nosso modelo classificará textos na área de ***política*** e ***automobilismo***. Você pode escolher qualquer uma das categorias ilustradas na imagem anterior, ou utilizar todas elas :-).

{% highlight python %}
categories = ['talk.politics.misc', 'rec.autos']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
{% endhighlight %}

## Treinando o classificador

Dados textuais representam valores discretos, e nosso classificador “só entende números”. Nós precisamos converter os dados brutos, que estão em formato de texto, para uma formato numérico. Isto deve acontecer antes de podermos passar os dados para o nosso classificador.

É preciso levar em conta, ainda, que algumas palavras no corpus de treino serão muito presentes, como é o caso de preposições e artigos. Estas palavras tendem a se repetir em todos os documentos e não costumam carregar informação muito significativa para o que precisamos aqui. Nós utilizaremos a medidade TF-IDF para limitar a importância destas palavras que se repetem muito ao longo dos documentos, de maneira que elas não causem mais influência do que o necessário. TF-IDF significa ***frequência do termo–inverso da frequência nos documentos*** e se baseia na seguinte formula.
{:.image}
![](https://cdn-images-1.medium.com/max/2608/1*V9ac4hLVyms79jl65Ym_Bw.png)

{% highlight python %}
vectorizer = TfidfVectorizer()
X_train_tfidf_vectorize = vectorizer.fit_transform(twenty_train.data)
{% endhighlight %}

Abaixo, nós iniciamos o processo de treino do nosso classificador, o que corresponderia a ajustar o estimador aos dados que nós temos. Iremos usar o ***MLPClassifier***, mas deixei o ***SGDClassifier***comentado, caso queira testar este estimador também. O ***MLPClassifier*** é, na verdade, uma rede neural Feed-Foward.

{% highlight python %}
# Aqui nós treinamos o classificador
#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70, ), random_state=1, verbose=True)
clf.fit(X_train_tfidf_vectorize, twenty_train.target)
{% endhighlight %}


O comando ***clf.fit()*** é responsável por treinar o nosso modelo.

## Avaliando a performance

Agora, nós precisamos avaliar a performance do nosso modelo para sabermos o quanto ele aprendeu com os dados de treino. Para isto, utilizaremos um subconjunto dos dados, os dados que separamos para os testes.

{% highlight python %}
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data

vect_transform = vectorizer.transform(docs_test)
predicted = clf.predict(vect_transform)


print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))

print(clf.classes_)
{% endhighlight %}

Em meus testes, eu obtive o seguinte resultado:

{:.image}
{:.zoom}
![*Avaliando a performance do modelo com algumas medidas de avaliação, como 'precision', 'medida-f1' e 'recall'*](https://cdn-images-1.medium.com/max/2000/1*aVNagC-IWUfDYp-Qk0iuKQ.png)*Avaliando a performance do modelo com algumas medidas de avaliação, como 'precision', 'medida-f1' e 'recall'*

## Matriz de Confusão

Usada para visualizar a performance de um classificador. As linhas da matriz indicam as classes que se espera obter corretamente por meio do modelo. As colunas indicam as classes que foram obtidas efetivamente. Cada célula contém o número de predições feitas pelo classificador, relativas ao contexto daquela célula específica.

{% highlight python %}
confusion_matrix = confusion_matrix(twenty_test.target, predicted)
print(confusion_matrix)

plt.matshow(confusion_matrix)
plt.title("Matriz de confusão")
plt.colorbar()
plt.ylabel("Classificações corretas")
plt.xlabel("Classificações obtidas")
plt.show()
{% endhighlight %}

A saída do código acima será algo como a seguinte figura:

{:.image}
![um exemplo de uma matriz de confusão para uma classificador binário](https://cdn-images-1.medium.com/max/2000/1*zjeRqyzr0MxzFhC4NI2row.png)*um exemplo de uma matriz de confusão para uma classificador binário*

## Fazendo predições em dados novos

Agora, chega a parte mais interessante, que é usar o nosso modelo para classificar textos que não estão presentes no conjunto de dados de treino, ou seja, textos que o modelo ainda “não viu”. Considerando que o dataset é composto de textos apenas em inglês, então, o nosso modelo não vai “entender outra língua”. É por isso que iremos testar o nosso modelo com textos em inglês:

{% highlight python %}
docs_new = [
    'Wednesday morning, the legal team had appeared to turn back toward more discreet lawyering, with the announcement that Washington trial lawyer Emmet Flood would join the team inside the White House.',
    'By the time Rolls-Royce unveiled its one-of-a-kind Serenity Phantom at the 2015 Geneva Motor Show.'
]

X_new_tfidf_vectorize = vectorizer.transform(docs_new)

predicted = clf.predict(X_new_tfidf_vectorize)

for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, twenty_train.target_names[category]))
{% endhighlight %}

O código acima irá exibir cada texto apresentado ao modelo por meio do array, com a categoria ao lado. Por exemplo, no meu experimento, eu obtive o seguinte retorno:

{:.image}
![Predição obtida por meio do modelo classificador de textos](https://cdn-images-1.medium.com/max/2000/1*Ypu6vhfHfcrc_8XoT4ngXA.png)*Predição obtida por meio do modelo classificador de textos*

Para mais detalhes e maiores explicações, veja o vídeo que está no início deste post. Aproveita e participa do[ grupo que criei no Slack](https://join.slack.com/t/falandosobreia/shared_invite/enQtMzI4NjkzMjI1NjgyLTZlN2VhN2VkMzc2MjYxNDYzOWZkNWEzZWQ5MmM5NDA2NWFmYzFlNTVjMmUxMzQyMGYwOTk4Y2JhNTI0ZjNmYzg), para tratar de assuntos diversos envolvendo machine learning e inteligência artificial.

E se você quiser saber um pouco mais sobre machine learning, veja o vídeo da live que apresentei recentemente, onde comentei sobre alguns aspectos teóricos desta que está sendo a profissão mais cobiçada da atualidade:

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/WgUrONLhons" frameborder="0" allowfullscreen></iframe></center>
</div>

## Leia também

1. [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

2. [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})

3. [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url blog/2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})