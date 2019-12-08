---
layout: post
title:  "Deep learning e Modelagem de Linguagem — Teoria e prática"
resume: "A Modelagem de Linguagem é um dos componentes mais importantes em processamento de linguagem natural, crucial em tarefas como tradução neural de textos."
date:   2018-12-26 11:50:00
categories: [nlp, deep-learning]
tags: [nlp, deep-learning]
permalink: /deep-learning/deep-learning-e-modelagem-de-linguagem-aprenda-na-pratica-com-tensorflow
thumbnail: "/blog/assets/img/deep-learning-e-modelagem-de-linguagem-aprenda-na-pratica-com-tensorflow.png"
---

{:.image}
![Deep learning e Modelagem de Linguagem — Teoria e prática](/blog/assets/img/deep-learning-e-modelagem-de-linguagem-aprenda-na-pratica-com-tensorflow.png)

{:.intro}
A modelagem de linguagem é um dos componentes mais importantes na área de processamento de linguagem natural, sendo uma peça fundamental em tarefas como **Machine Translation** e **Reconhecimento de voz**. Por esta razão, é uma área de pesquisa que recebe bastante atenção frequentemente. Mas o que seria isso, exatamente?

# O que significa Modelagem de Linguagem?

Para ser sucinto, trata-se de atribuir probabilidades às sentencas em um texto. Por exemplo, considere o seguinte questionamento: 

> Qual seria a probabilidade de vermos a frase: **minha terra tem palmeiras onde canta o sabiá**?

Além disso, o modelo também tenta atribuir uma probabilidade para cada palavra que aparece numa sequência de palavras presentes na mesma frase, parágrafo ou texto. Considere agora este outro questionamento:

> Qual seria a probabilidade de vermos a palavra **sabiá**, dado que temos a frase **minha terra tem palmeiras onde canta**?

De uma forma geral, o modelo tenta adivinhar qual é a próxima palavra a aparecer na frase, diante de um conjunto de palavras que já existem nela. Obviamente você já viu isto funcionar na prática enquanto digitava alguma coisa no seu smartphone ou tablet e cada palavra que você queria usar na sequência ia aparecendo na parte superior do teclado virtual. 

{:.image}
![modelágem de linguagem](/blog/assets/img/modelagem-de-linguagem-teclado-ios.jpg)
*o teclado virtual do seu smartphone utiliza modelagem de linguagem para estimar qual é a próxima palavra que você quer digitar, baseado no que você já digitou antes.*

Neste ponto, você já deve ter notado que nós podemos adotar este tipo de abordagem para gerar sentenças, parágrafos e até mesmo livros inteiros.

# Matematicamente, como isto poderia ser representado?

Para expressar matematicamente o que já foi dito aqui, a tarefa de modelar linguagens é atribuir uma probabilidade à qualquer sequência de palavras $$w_{1:n}$$, ou seja, estimar $$P(w_{1:n})$$. Pela regra da cadeia da probabilidade é mais fácil expressar isto:

{:.block}
$$
   P(w_{1:n}) = P(w_1)P(w_2 \mid w_1)P(w_3 \mid w_{1:2})P(w_4 \mid w_{1:3})... P(w_n \mid w_{1:n-1}) 
$$

Isto significa predizer palavras sequencialmente onde a estimativa de cada palavra está condicionada às palavras precedentes. Modelar uma única palavra com base no contexto presente à esquerda dela é preferível em relação à atribuir um score para uma frase inteira. 

Porém, se pensarmos no fato de que o último termo na sequência da equação anterior $$ P(w_n \mid w_{1:n-1}) $$ ainda está condicionado à todas as $$ n-1 $$ palavras, nós vemos que há um problema nisto, porque este termo é tão chato de modelar quanto uma frase inteira. 

Durante muito tempo os modelos de linguagem fizeram uso da [Propriedade de Markov](https://en.wikipedia.org/wiki/Markov_property) para contornar este problema, a qual declara o seguinte:

> A distribuição de probabilidade condicional de estados futuros do processo depende apenas do estado atual.

Aplicando isto no contexto do nosso problema de modelagem de linguagem, nós podemos concluir que **a próxima palavra em uma sequência depende apenas das últimas *k* palavras**:

{:.block}
$$
   P(w_{i+1} \mid w_{1:i}) \approx P(w_{i+1} \mid w_{i-k:i}).
$$

Assim, a estimativa da probabilidade da senteça pode ser representada desta forma:

{:.block}
$$
   P(w_{1:n}) \approx \prod_{i=1}^{n}P( w_i \mid w_{i-k:i-1})
$$

O objetivo, então, é precisamente estimar $P(w_{i+1} \mid w_{i-k:i})$ com base em um conjunto de dados de texto. Quanto maior for o volume de dados que você tiver à disposição, mais preciso será o seu modelo. 

Acontece que este método acima possui uma fraqueza nítida. Repare que este cálculo é baseado em um **produto**, de maneira que se houver um termo qualquer que não tenha sido observado no conjunto de aprendizagem, então a probabilidade dele será **0** (zero) e isto anulará toda a equação. Contudo, algumas técnicas foram utilizadas para contornar este problema, impedindo que os valores se anulem. Mesmo assim, há um limite para o que podemos fazer com esta abordagem. Apesar de ela produzir resultados interessantes em pequenos conjuntos de dados, ela costuma falhar em sequências maiores e mais complexas com dependências distantes que são mais difíceis de modelar (contextos maiores).

{:.note}
Modelos tradicionais de linguagem baseados em propriedades de Markov costumam falhar em sequências maiores e mais complexas com dependências distantes que são mais difíceis de modelar.

# Modelos Neurais de Linguagem e ***Word Embeddings***

Modelos neurais de linguagem resolvem muitos dos problemas que são encontrados nas abordagens tradicionais. Redes neurais têm algumas propriedades bem convenientes no âmbito da modelagem de linguagem, apesar do seu custo proibitivo diante de conjuntos de dados maiores em muitas ordens de magnitude. Uma destas vantagens é poder administrar contextos mais amplos de uma forma muito mais eficiente. Isto acontece, principalmente, nas arquiteturas de redes neurais recorrentes mais modernas, tais como a LSTM e a GRU ([Cho et al. 2014](https://arxiv.org/abs/1406.1078)).

Quando você usa redes neurais para modelar linguagens, é útil criar representações de palavras distribuídas em espaços vetoriais, onde cada palavra é representada como um vetor de valores reais (nada de símbolos discretos aqui). O significado de cada termo e sua relação com outras entidades são capturados pelas ativações presentes neste vetor que representa cada palavra, bem como pelas similaridades entre os diferentes vetores. 

{:.image}
![Modelagem de linguagem - word Embeddings](/blog/assets/img/deep-learning-e-modelagem-de-linguagem-word-embeddings.png)
*Vetores de palavras onde cada palavra é representada como um vetor em um espaço de baixa dimensionalidade (cada linha na tabela acima é um vetor). As dimensões mais comuns para este vetor são: 50, 100, 300, 800 e 1000.*


Estas representações de palavras em espaço vetorial são normalmente chamadas de ***Word embeddings*** na literatura e podem "alimentar" redes neurais para processamento de linguagem natural, melhorando suas capacidades.

{:.note}
Nós podemos representar cada feature (palavra) como um vetor em um espaço de baixa dimensionalidade. Na prática, isto significa que cada palavra é mapeada para um vetor de valores reais *d*-dimensional e o significado da palavra será capturado pela sua relação com outras palavras e seus valores de ativação presentes em seus respectivos vetores.

Abordagens como as de [**Mikolov et al., 2013b**](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) e [**P. Bojanowski et al., 2016**](https://arxiv.org/abs/1607.04606) se popularizaram recentemente porque trouxeram melhorias significativas nas performances de modelos neurais para processamento de linguagem natural. Trata-se de modelos semânticos de linguagem, obtidos a partir treinos não-supervisionados em grandes corpus de textos não estruturados, que se baseiam na seguinte premissa: as palavras que compartilham significados possuem vetores muito similares porque também foram utilizadas em contextos similares. Assim, estas palavras podem ser agrupadas muito próximas umas das outras numa mesma região do espaço vetorial.

{:.image}
![Word Embeddings](/blog/assets/img/word_embeddings_most_similar_2.png)
*palavras que foram usandas em contextos similares possuem vetores similares e são agrupadas numa mesma região do espaço vetorial, justamente por compartilharem algum significado. Para saber como plotar um gráfico assim, veja [este link](https://www.kaggle.com/luisfredgs/plotando-gr-fico-de-word-embedding)*

Estas *word embeddings* vão entrar em uma camada da rede normalmente referenciada como ***Embedding Layer***. Você até pode inicializar esta camada com pesos aleatórios e depois ajustar estes pesos progressivamente durante o treino até encontrar uma combinação ótima. Mas, melhor ainda do que isso é já poder inicializá-la utilizando estas representações de palavras previamente obtidas com base no vocabulário do seu conjunto de aprendizagem, ou até mesmo com base em vetores de palavras pré-treinados em dados de outros domínios.

Alguns destes vetores de palavras são treinados em grandes corpus de dados não supervisionados, como a Wikipédia. Treinar *word embeddings* usando estas abordagens em quantidades de dados tão grandes assim pode trazer importantes benefícios, como o de fornecer representações vetoriais para palavras raras, que normalmente nem aparecem no seu conjunto de aprendizagem ([**P. Bojanowski et al., 2016**](https://arxiv.org/abs/1607.04606)). Teoricamente, as representações para estas palavras serão similares àquelas de palavras relacionadas que estariam no conjunto de aprendizagem, permitindo que o modelo obtenha melhores generalizações.

Considerando que as palavras são representadas como vetores, o modelo pode computar as similaridades entre estas palavras ao calcular as similaridades entre os vetores. Duas medidas de similaridade comuns são a *cosine* e *Jacaard Similarity*. A *cosine* mede o cosseno do ângulo entre os vetores $u$ e $v$ (digamos que $u$ represente uma palavra e $v$ represente uma ourta palavra):

{:.block}
$$
sim_{cos}(u, v) = \frac{u \cdot v}{ \left\Vert u \right\Vert_2 \left\Vert v \right\Vert_2 } = \frac{ \sum_{i}u_{[i]} \cdot v_{[i]} }{ \sqrt{ \sum_{i}(u_{[i]})^2 } \sqrt{ \sum_{i}(v_{[i]})^2 } }
$$

A *Jacaard Similarity* é definida assim:

{:.block}
$$
sim_{Jacaard}(u, v) = \frac{\sum_{i}min(u_{[i]}, v_{[i]})}{\sum_{i}max(u_{[i]}, v_{[i]})}
$$

Para ilustrar esta medida de similaridade, vamos extrair os vetores de 3 palavras presentes [neste word embedding](http://143.107.183.175:22980/download.php?file=embeddings/word2vec/cbow_s50.zip), com vetores em 50 dimensões obtidos por meio de [**Mikolov et al., 2013b**](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) e [**P. Bojanowski et al., 2016**](https://arxiv.org/abs/1607.04606). As palavras são: ***presidente***, ***estado*** e ***comissão***. Vamos calcular as similaridades entre elas:

````python
>>> import numpy as np
>>> from sklearn.metrics.pairwise import cosine_similarity

>>> # u = presidente
>>> u = [-0.093907, 0.271813, 0.045385, -0.062937, 0.135744, -0.495446, -0.633432, -0.278358, -0.281258, -0.352969, -0.389262, 0.037469, -0.576813, -0.106252, -0.131994, 0.037046, 0.151033, -0.178932, -0.449697, -0.074721, -0.643320, 0.471278, -0.275966, 0.350979, 0.285883, 0.280474, -0.281239, 0.363799, 0.159182, 0.434006, 0.020838, -0.398882, 0.339602, -0.055495, -0.196537, 0.323546, 0.080960, 0.033596, 0.265669, 0.438394, -0.287924, -0.161542, -0.051969, -0.266580, 0.080336, -0.073548, -0.224689, -0.007176, 0.022561, -0.229548]

>>> # v = estado
>>> v = [0.254259, 0.588025, -0.056152, -0.055573, 0.206882, 0.408681, -0.325860, -0.144961, 0.671040, -0.237184, -0.009760, 0.785549, 0.249680, -0.018667, 0.296448, -0.005555, 0.163902, -0.462612, -0.203502, -0.152195, -0.898882, 0.567912, 0.136436, 0.785681, 0.137562, 0.366535, -0.104239, 0.164342, 0.289210, 0.004095, 0.027409, -0.560103, 0.997733, -0.346338, 0.505060, 0.267630, 0.159966, -0.278708, 0.134177, 0.344001, -0.505864, -0.431534, -0.403217, 0.218687, -0.011242, 0.524582, -0.333766, 0.035323, -0.136235, -0.627216]

>>> # w = comissão
>>> w = [-0.306975, 0.379277, -0.014631, 0.026234, 0.167269, -0.062476, -0.224295, -0.224618, -0.303858, 0.156032, -0.360141, -0.185232, 0.091300, 0.069819, -0.402067, 0.574204, -0.274878, -0.237076, -0.605752, 0.312052, -0.785056, -0.082266, -0.354681, 0.137450, -0.120481, -0.018686, 0.217644, -0.069758, 0.102755, -0.008425, 0.175850, -0.435626, 0.085948, -0.659927, -0.050797, 0.380319, -0.394668, -0.182752, 0.502834, 0.170226, 0.289859, 0.456589, 0.248910, 0.032968, -0.264588, -0.392039, -0.495116, -0.057584, -0.341951, 0.204977]

>>> # similaridade entre a palavra presisente e estado
>>> print(cosine_similarity(np.array([u]), np.array([v])))

[[0.45978313]]

# similaridade entre a palavra presidente e comissão
>>> print(cosine_similarity(np.array([u]), np.array([w])))

[[0.38518093]]

````
A medida de similaridade baseada no cosseno do ângulo entre os dois vetores nos dá um resultado entre **0** (zero) e **1** e quanto mais próximo de **1** for o resultado, mais similares são os vetores e, portanto, mais similares serão os dois termos. Pela saída emitida no código acima, nós podemos perceber que os termos **presidente**  e **estado** (cosseno = 0.45978313) são mais similares do que **presidente** e **comissão** (cosseno=0.38518093), ao menos com base no contexto em que elas foram utilizadas no dataset.

# Exemplo prático

Foi mencionado no início deste post que modelos de linguagem podem ser usados para gerar sentenças. Isto, é claro, depois de treinados em algum conjunto de aprendizagem. Eu desenvolvi um pequeno código para exemplificar os conceitos abordados aqui neste post e você pode usá-lo como ponto de partida nos seus estudos. O vídeo abaixo mostra este código funcionando no Google Colab.

<div class="video-container">
	<iframe width="560" height="315" src="https://www.youtube.com/embed/riS1Zf1cgYk?rel=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Para exemplificar o uso de **Word embeddings** pré-treinados, eu inicializei a camada embedding com word vectors *300*-dimensional obtidos por meio de um modelo não-supervisionado treinado em um [dump de 2017 da Wikipedia em inglês](https://fasttext.cc/docs/en/english-vectors.html). Veja um trecho do código apenas para ilustrar:

````python

# Esta função nos permite usar um Word Embedding pré-treinado
# em um outro conjunto de dados. Como nosso dataset nesta tarefa é
# muito pequeno, pode ser uma boa ideia usar um word embedding pré-treinado

def load_fasttext(word_index, max_features):    
    embeddings_index = {}
    f = codecs.open("wiki.pt.vec", encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('%s word vectors encontrados' % len(embeddings_index))
    
    words_not_found = []
    
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # palavras nao encontradas no embedding permanecem com valor nulo
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    
    return embedding_matrix
````

Agora, nós carregamos os word vectors em uma variável e alimentamos a ***Embedding Layer*** com estes vetores:

````python

# carrega nosso word embedding pré-treinado
embedding_matrix = load_fasttext(word_index, max_features)

model = Sequential()
model.add(Embedding(
	max_features, 
	300, 
	weights=[embedding_matrix], # pesos do word embedding pré-treinado 
	input_length=max_length-1, 
	trainable=False))
model.add(Bidirectional(GRU(50)))
model.add(Dense(vocab_size, activation='softmax'))

````

Perceba que, como estes pesos já foram treinados, nós setamos a opção **trainable=False**. Porém, você pode gerar estes word embeddings com base no vocabulário do seu dataset e utilizá-los para inicializar esta mesma camada. Isto é preferível, dependendo da situação. Como nosso dataset é muito pequeno, acabei usando um pré-treinado mesmo.

Postei todo o código [no github](https://github.com/luisfredgs/language-modeling-tensorflow-keras) com algumas instruções para que você possa reproduzir os resultados no Google Colab ou na sua máquina.

Dúvidas? Comentários? Sugestôes? Escrevam aí em baixo, na área de comentários! :-)

## Leia também

1. [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

2. [Dicas para aprender Machine Learning]({% post_url 2018-01-18-dicas-para-aprender-machine-learning %})

3. [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

4. [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})

5. [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})
