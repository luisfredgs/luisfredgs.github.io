---
layout: post
title:  "Análise de sentimentos com redes neurais recorrentes LSTM"
resume: "Aprenda, na teoria e na prática, como fazer análise de sentimentos com redes neurais artificiais. Aprenda como funcionam as redes neurais recorrentes LSTM e como aplicá-las na criação do seu próprio modelo para classificar sentimentos, usando tensorflow/keras. Este post acompanha um vídeo no final, que mostra teoria, código e o modelo funcionando ao vivo"
date:   2018-12-23 17:50:00
categories: [nlp, deep-learning]
tags: [nlp, deep-learning]
permalink: "/deep-learning/analise-de-sentimentos-usando-redes-neurais"
status: 1
---
{:.image}
![](/assets/img/analise-de-sentimentos-usando-redes-neurais.png)

{:.intro}
*Aprenda, na teoria e na prática, como fazer análise de sentimentos com redes neurais artificiais. Aprenda como funcionam as redes neurais recorrentes LSTM e como aplicá-las na criação do seu próprio modelo para classificar sentimentos, usando tensorflow/keras. Este post acompanha um vídeo no final, que mostra teoria, código e o modelo funcionando ao vivo*

A análise de sentimentos tornou-se uma ferramenta fundamental para diversas necessidades no mundo corporativo. Seja para uso no monitoramento de reputação online ou como componente de chatbots, para alertar operadores humanos ao menor sinal de descontentamento do cliente durante o fluxo conversacional. Trata-se de um problema de classificação de textos em linguagem natural que pode muito bem ser resolvido com machine learning usando modelos baseados em SVM, Naive Bayes, Esemble ou até mesmo Regressão.

De fato, há ocasiões em que os algoritmos mais simples conseguem mesmo ter performance melhor (como é o caso do SVM), mas é algo que depende muito de como você monta a sua rede, ou da quantidade de dados que você possui para a aprendizagem do modelo, ou de quanto ruído há nestas informações. Mas, no geral, os modelos baseados em deep learning costumam se sair bem melhor.

## Porque Redes Neurais Recorrentes?

As redes neurais recorrentes (RNR ou RNN — Recurrent Neural Networks) são uma família de redes neurais voltadas para o processamento de dados sequenciais. Portanto, constituem-se num recurso bastante apropriado em situações onde há a necessidade de classificar textos. Uma situação muito comum onde estas arquiteturas costumam ser utilizadas é quando surge a necessidade de predizer a próxima palavra a fazer parte de uma sentença, no momento em que você está escrevendo um texto. Isto acontece quando você está digitando algo no teclado do seu smartphone para enviar uma mensagem pelo WhatsApp, por exemplo, ou quando está enviando um e-mail.

Uma variante das RNNs convencionais é a [LSTM](http://www.bioinf.jku.at/publications/older/2604.pdf) (Long Short-Term Memory), que resolve um problema conhecido chamado [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), o qual ocorre quando os pesos computados nas partes iniciais da sequência perdem influência ao longo das iterações, sendo sensíveis às novas entradas. Com isto, a variação de informações contextuais que pode ser captada pelas RNNs comuns costuma ser bastante limitada. Tal limitação leva estas arquiteturas a ter uma performance bastante sofrível em sequências mais longas.

A LSTM surgiu para resolver este problema, sendo uma RNN com subestruturas que ajudam a gerenciar a “memória” da rede neural recorrente. Estas subestruturas são denominadas *células de memória,* e são fundamentais na solução do problema que a LSTM se propões a resolver. Assim, sendo a análise de sentimentos uma tarefa de classificação de textos, é natural a escolha das RNNs nestes casos.

## Representando palavras em espaços vetoriais

Ferramentas de processamento de linguagem natural (NLP — Natural Language Processing) baseadas em machine learning precisam que as palavras de um texto estejam representadas em formato vetorial e numérico, considerando que símbolos discretos não carregam informação útil quando o interesse é usar o computador para interpretar linguagem natural.

Ao criar uma representação das palavras em um espaço vetorial contínuo, nós podemos codificar diversas regularidades linguísticas, tais como os relacionamentos semânticos presentes nos textos, de maneira que as palavras que possuem um significado similar são mapeadas em pontos muito próximos dentro deste espaço de vetor.

Esta proximidade, que pode ser calculada usando aritmética vetorial, reflete a similaridade entre duas ou mais palavras. Normalmente se utiliza a [distância cosseno](https://en.wikipedia.org/wiki/Cosine_similarity) ou [euclidiana](https://pt.wikipedia.org/wiki/Dist%C3%A2ncia_euclidiana) entre dois vetores para calcular esta similaridade. No caso da distância cosseno, quanto mais próximo de **1** for o resultado, mais similares são duas palavras dentro deste espaço.

{:.image}
![Cálculo da distância cosseno](https://cdn-images-1.medium.com/max/2000/1*kQiCiH39yc09V6ZaSjC7QQ.png)
*Cálculo da distância cosseno*

Uma abordagem que se tornou bastante comum para computar este tipo de representação é a [Word2vec](https://arxiv.org/pdf/1301.3781.pdf) (Mikolov et al, 2013). Na figura abaixo, há uma plotagem de um word embedding desses, que ilustra a utilidade desta técnica. Veja que, palavras que possuem um significado similar estão agrupadas muito próximas dentro do espaço de vetor.

{:.image}
![Word Embedings representam palavras em espaço vetorial contínuo que codifica diversas regularidades linguísticas.](https://cdn-images-1.medium.com/max/2118/1*GemG0fQLn8EPIQz-IsAXWw.png)
*Word Embedings representam palavras em espaço vetorial contínuo que codifica diversas regularidades linguísticas.*

Palavras que aparecem em um mesmo contexto, costumam compartilhar um significado semântico. Os Word Embeddings codificam justamente estas regularidades. É por isso que esta técnica costuma ser tão útil em tarefas de classificação de textos envolvendo NLP.

Para saber como plotar o gráfico exibido acima, você pode visualizar e rodar o [código deste link](https://www.kaggle.com/luisfredgs/plotando-gr-fico-de-word-embedding?scriptVersionId=5666020), no site da Kaggle. Trata-se de um kernel que mostra como importar word vectors baseados em Word2Vec e plotar o gráfico.

## Análise de sentimentos usando Redes Neurais Recorrentes LSTM e o Tensorflow por meio do Keras

Se tudo o que foi dito até agora só te deixou com mais dúvidas, então eu recomendo que veja este vídeo que gravei recentemente, onde explico (na teoria e na prática) o funcionamento das Redes Neurais Recorrentes e a variante LSTM em mais detalhes, entre outros conceitos.

O vídeo mostra como você pode criar o seu próprio modelo para classificar sentimentos com base em redes LSTM e no framework tensorflow, por meio da lib Keras. Também mostro o modelo funcionando ao vivo, em textos que escolho aleatoriamente na seção de reviews da amazon, para testar a capacidade de predição.

Quando você divide seus dados em treino/teste, você na verdade está “sacrificando” informação que deveria expor ao seu modelo para que ele obtenha melhor performance de generalização. Se você vai usar o modelo em produção, seria desperdício deixar de fora aquela parcela dos dados que reservou para testes. Então, aí vai uma dica que nunca encontrei nos livros…

**Se você vai rodar seu código em produção**:

1. particione seus dados em ‘treino’ e ‘teste’, como sempre faz.

1. Treine o seu modelo no conjunto de aprendizagem normalmente e tune os hiper parâmetros com base no conjunto de testes.

1. Depois de tunar os hiperparâmetros com base no conjunto de teste e tendo encontrado a combinação ideal, treine novamente o seu modelo **no conjunto de dados inteiro**, sem particiona-los (se você já descobriu que seu modelo performa bem nos dados de teste, então há grandes chances de o mesmo acontecer no conjunto de dados inteiro). Na prática, isto aumenta a quantidade de dados que seu modelo usará para “aprender” e isto te dará algum percentual a mais em performance no modelo final, algo que pode fazer toda uma diferença quando você possui dados muito escassos. Depois disso, exporte o seu modelo e rode-o em produção.

Não permita que o valor da acurácia no conjunto de testes se distancie muito daquele valor obtido no conjunto de testes para a mesma métrica. Se isto acontecer, significa que você está diante de um overfitting e você vai precisar contornar isto (normalmente por meio de técnicas de reguralização, como Dropout, um bom tratamento nos dados, experimentar um otimizador diferente, etc)

Eu compartilhei o [código no github](https://github.com/luisfredgs/sentiment-analysis-keras-lstm), então vocês podem reproduzir os resultados. **Caso queiram rodar o código em uma versão traduzida do dataset IMDb para o português**, vocês podem fazer um fork [deste kernel na Kaggle e executar o código lá mesmo](https://www.kaggle.com/luisfredgs/imdb-dataset-em-portugu-s-e-ingl-s), com GPU Free (Eles disponibilizam GPUs Nvidia Tesla K80, gratuitamente). Eu recomendo que vejam o vídeo primeiro (abaixo), caso não estejam confortáveis ainda com a teoria por trás da redes neurais recorrentes.

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/bIcadBu--u8" frameborder="0" allowfullscreen></iframe></center>
</div>

## Leia também

1. [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

2. [Dicas para aprender Machine Learning]({% post_url 2018-01-18-dicas-para-aprender-machine-learning %})

3. [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

4. [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})
