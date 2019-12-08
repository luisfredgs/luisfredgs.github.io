---
layout: post
title:  "O que é Sequence-to-sequence em deep learning?"
resume: "Mesmo que você só tenha começado a estudar machine learning há pouco tempo, e ainda não saiba o que é sequence-to-sequence (seq2seq), é quase certo de que já tenha ouvido falar neste termo. Trata-se de uma metodologia baseada em redes neurais que está presente no core de muitas aplicações que usamos hoje em dia."
date:   2018-04-10 11:50:00
categories: [nlp, deep-learning]
tags: [nlp, deep-learning]
permalink: /deep-learning/o-que-e-sequence-to-sequence-em-deep-learning
---

{:.image}
![](/blog/assets/img/o-que-e-sequence-to-sequence-em-deep-learning.png)

Mesmo que você só tenha começado a estudar machine learning há pouco tempo, e ainda não saiba o que é sequence-to-sequence (seq2seq), é quase certo de que já tenha ouvido falar neste termo. Trata-se de uma metodologia baseada em redes neurais que está presente no core de muitas aplicações que usamos hoje em dia, e que se baseiam em inteligência artificial. Entre os mais importantes casos de uso desta metodologia, estão o Google Search e o Google Tradutor. Há também outras aplicações interessantes, como casos de uso em modelagem de agentes conversacionais.

Ao final deste post, você terá lido sobre:

1. O que é Sequence-to-Sequence

1. Como isto pode ser útil? Quais são as aplicações no mundo real?

1. Como as redes neurais são utilizadas na abordagem Sequence to Sequence?

1. A arquitetura Encoder-Decoder baseada em Redes Neurais Recorrentes (RNN)

1. O mecanismo Attention

1. O que é a arquitetura Transformer

*Antes de prosseguir, [participe do novo grupo no Slack](https://blog.luisfred.com.br/falando-sobre-inteligencia-artificial-novo-grupo-no-slack/) que criei há pouco tempo para promover conversas sobre Inteligência artificial, Machine Learning e Deep Learning.*

## O que é Sequence-to-sequence?

Mas, afinal, o que é Sequence-to-sequence? Qual seria a teoria por trás disso e quais são as principais aplicações? Na verdade, Seq2Seq é a abordagem na qual se utiliza modelos de machine learning / deep learning para obter uma sequência como entrada, em um domínio específico, e converter esta sequência para uma representação em outro domínio. É trazer uma distribuição de probabilidades de sequências de saída e maximizar a probabilidade de uma sequência alvo, dada uma determinada sequência anterior como entrada. Tome como exemplo, a tradução de idiomas e você terá uma visão do quão importante esta abordagem tem se tornado. Na prática, acontece assim:

*“le chat est noir” -> **[Seq2Seq]** -> “the cat is black”*

Por baixo do capô, é da seguinte forma que funciona:

{:.image}
![Sequence to Sequence](https://cdn-images-1.medium.com/max/2000/0*tDxEP2_zyt79i3Ro.png)

## Como isto pode ser útil? Quais são as aplicações no mundo real?

Na tradução neural de máquinas (Neural Machine Translation ), encontramos alguns exemplos de uso, não apenas no ambiente de pesquisa, mas em produção também (Google e Microsoft, por exemplo). Sem dúvidas, esta é uma das aplicações mais interessantes do Seq2Seq. Mas não se restringe apenas à estes casos pois, além disso, pode-se aplicar esta técnica em:

1. [Sistemas de pergunta e resposta](https://en.wikipedia.org/wiki/Question_answering), que respondem automaticamente às perguntas feitas por pessoas

1. [Aprendizado de agentes conversacionais](https://arxiv.org/pdf/1506.05869v1.pdf), como os chatbots

1. Criação de bases de conhecimento

1. Inteligências artificiais que interpretam textos ([Machine Reading Comprehension](https://www.microsoft.com/en-us/research/blog/transfer-learning-machine-reading-comprehension/))

1. [Tradução neural de máquina](https://research.google.com/pubs/pub45610.html)

## Como as redes neurais são utilizadas na abordagem Sequence to Sequence?

Vou usar o Google Tradutor como referência aqui, porque é uma das melhores formas de se ver o potencial de toda esta ideia. Você já reparou que a precisão do Google Tradutor melhorou muito de uns tempos para cá? A ferramenta consegue captar muito melhor o contexto das frases e nos trazer traduções bem melhores do que já proporcionara até pouco tempo atrás. Isto se deve à aplicação de redes neurais que deu origem à [segunda geração do tradutor deles](https://blog.google/products/translate/higher-quality-neural-translations-bunch-more-languages/), que estamos utilizando hoje. Acontece que eles não utilizam redes neurais comuns, como seria o caso das redes neurais [*feed-foward*](https://en.wikipedia.org/wiki/Feedforward_neural_network) (alimentadas adiante). Isto porque estas arquiteturas mais básicas de redes neurais não trariam uma boa performance no mapeamento de sequências. Neste caso, diferentes arquiteturas são combinadas até que se obtenha um modelo satisfatório.

Todas estas aplicações de machine learning atualmente funcionando em produção dentro das grandes empresas, ao menos uma boa parte delas, partiram de iniciativas na pesquisa acadêmica. Eles estudam, fazem experimentos, comparam resultados e constatam sua viabilidade técnica para uso em larga escala. É assim com a Google, Microsoft, Facebook, IBM, etc. [**Sutskever et al.**](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), 2014 publicou um dos primeiros papers acadêmicos que mostra a abordagem Seq2Seq como sendo viável para fazer tradução de textos (com base em Neural Machine Translation).

Com um vocabulário composto por 160k palavras em inglês, dado como entrada para um modelo, os autores usaram uma LSTM (Long Short-Term Memory) para representar uma sentença de entrada em 8k números reais (valores contínuos). Estes 8k valores contínuos correspondem a uma representação intermediária, agrupados em espaço vetorial de tamanho fixo. Uma segunda rede LSTM utiliza estes valores como entrada para produzir sentenças em Francês a partir deste vetor intermediário, também chamado de *vetor de contexto*. Os autores utilizaram redes LSTM de 4 camadas. Esta abordagem que eles adotaram é conhecida como Encoder-Decoder.

## A arquitetura Encoder-Decoder baseada em Redes Neurais Recorrentes (RNN)

De uma forma resumida, o trabalho de **sutskever et al. 2014** contou com duas LSTMs multicamadas. Uma para mapear uma sequência de entrada em um vetor intermediário de dimensionalidade fixa. Em seguida, vem uma outra LSTM que decodifica uma sequência de saída a partir deste vetor. Estes componentes são treinados em conjunto, para maximizar a probabilidade de uma tradução correta, dada uma sentença de origem.

> *A idéia é usar uma LSTM para ler a seqüência de entrada, um timestep por vez, para obter uma grande representação vetorial de dimensões fixas, e em seguida usar outra LSTM para extrair a sequência de saída a partir daquele vetor.* - [“Sequence to Sequence Learning with Neural Networks,”](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) 2014

![O modelo lê uma sentença de entrada “ABC” e produz “WXYZ” como a sentença de saída. Ele deixa de fazer previsões depois de gerar o token de fim de frase “EOS”- **sutskever et al. 2014**](https://cdn-images-1.medium.com/max/2048/0*1-FEHjs7oLced7d1.png)*O modelo lê uma sentença de entrada “ABC” e produz “WXYZ” como a sentença de saída. Ele deixa de fazer previsões depois de gerar o token de fim de frase “EOS”- **sutskever et al. 2014***

Trata-se de uma metodologia bastante recente, tendo se tornado mais conhecida principalmente depois que a [Google a adotou](https://research.google.com/pubs/pub45610.html) como tecnologia principal no seu sistema de tradução de textos. Eles obtiveram grandes resultados ao testar esta abordagem em datasets mais famosos no que diz respeito a este domínio de tradução. Eles usaram o [WMT’14](http://www.statmt.org/wmt14/translation-task.html) no benchmark.

> *[…] nossa abordagem é transferida para conjuntos de dados de produção muito maiores, que têm várias ordens de magnitude a mais de dados, para fornecer traduções de alta qualidade.* - [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), 2016

Após isto, surgiram muitas outras aplicações envolvendo não apenas modelos de tradução, mas também em outras situações, como modelagem de agentes conversacionais (pode chamar de chatbots, se quiser). Neste caso, isto funcionaria como algo próximo do seguinte:

{:.image}
![Um exemplo de modelagem de agentes conversacionais usando redes neurais recorrentes LSTM](https://cdn-images-1.medium.com/max/2000/0*ODdjm7CjCnfbldGg.png)*Um exemplo de modelagem de agentes conversacionais usando redes neurais recorrentes LSTM*

## O mecanismo Attention

A arquitetura Encoder-Decoder, entretanto, traz um grande inconveniente quando surgem sequências grandes. [**Cho, et al.**](https://arxiv.org/pdf/1409.1259.pdf) e [**Bahdanau, et al.**](https://arxiv.org/pdf/1409.0473.pdf) observaram que a performance dos modelos baseados nesta arquitetura se degrada conforme aumenta o tamanho das sentenças.
> *Uma limitação em potencial que surge com a abordagem encoder-decoder é que a rede neural precisa ser capaz de condensar toda a informação necessária da sentença de entrada em um vetor de dimensões fixas. Isto impede que a rede consiga lidar com sequências mais longas, especialmente quando o modelo se depara com sequências maiores do que aquelas existentes no conjunto de dados de treino.*  - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), bahdanau et al. 2014.

Para contornar este problema, foi proposta uma extensão do encoder-decoder na forma de um mecanismo denominado **Attention**, no paper “[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)”. A ideia é simples, porém brilhante. Baseia-se no princípio de que, no lugar codificar toda a sentença de entrada em um vetor de contexto de tamanho fixo, o modelo “presta atenção” em partes específicas da sentença de entrada, nas quais as informações mais relevantes estejam concentradas. Este modelo, então, obtém a predição da palavra-alvo baseado no contexto presente no vetor intermediário, associado com estas partes específicas da sentença de entrada, bem como todas as palavras geradas em *steps* anteriores.
> *A mais importante diferença entre esta abordagem e a arquitetura encoder-decoder básica é que ela não tenta codificar toda a sentença de entrada em um simples vetor de tamanho fixo.* - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), bahdanau et al. 2014.

O modelo de [**Bahdanau, et al.**](https://arxiv.org/pdf/1409.0473.pdf) também se baseia em redes neurais recorrentes, mas no lugar de usar LSTM, eles usaram a arquitetura [GRU](https://arxiv.org/pdf/1406.1078v3.pdf) **(Cho, et al., 2014).** Além disso, o modelo traz uma técnica genial que promove um alinhamento entre a sentença de entrada e a sentença de saída correspondente.

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*cjkzXnJQCagZFZoA.png)
*Os eixos x e y de cada gráfico correspondem às palavras na sentença de origem (Inglês) e a tradução obtida pelo modelo (Francês), respectivamente. As diferentes intensidades dos pixels mostram o quanto uma palavra na sentença de origem corresponde com outra na sentença alvo. Isto é usado como critério para alinhar as sentenças antes de obter a tradução.*

E se você quiser **meter a mão no código** e testar esta implementação em algum framework de deep learning, basta[ acessar este tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) na página de documentação do framework [Pytorch](http://pytorch.org/). Trata-se de um dos frameworks de deep learning “mais lindos” que eu já vi, sendo muito indicado para o ambiente de pesquisas. O Pytorch é bem menos verboso do que o [tensorflow](https://www.tensorflow.org/) e isto ajuda na interpretação das implementações só de bater o olho no código.

## A arquitetura Transformer

Redes neurais recorrentes costumam ser computacionalmente intensivas, dada a sua natureza sequencial. A rigor, quanto mais informação a RNN precisar processar até captar o contexto de uma sequência, mais recursos computacionais ela irá consumir. Ao mesmo tempo, isto se torna uma barreira ao aproveitamento eficaz da paralelização em GPUs por parte destas arquiteturas. É algo que vem a ser mais crítico em sequências longas e em datasets maiores.

Técnicas mais recentes proporcionaram melhorias significativas com respeito à eficiência computacional e na performance dos modelos baseados nestas arquiteturas recorrentes, como é o caso do **Attention**. Mesmo assim, estes métodos ainda não aproveitam bem a paralelização em GPUs e carecem, portanto, de performance. [Redes neurais Convolucionais](https://blog.luisfred.com.br/reconhecimento-de-escrita-manual-com-redes-neurais-convolucionais/) (CNNs) são menos sequenciais do que as RNNs mas, ainda assim, o número de passos necessários para computar informações de longo prazo em uma sequência aumenta com a distância entre estas informações. Isto ainda pode comprometer bastante a performance e a eficiência computacional destes modelos.

[**Vaswani et al. (2017)**](https://arxiv.org/abs/1706.03762) propôs uma nova arquitetura neural que evita completamente esta dependência das RNNs, e também CNNs. No lugar disto, eles contaram apenas com um mecanismo [*self-attention*](https://arxiv.org/pdf/1703.03130.pdf) e redes feed-foward. O novo método, denominado **Transformer**, é [computacionalmente menos exigente](https://research.googleblog.com/2017/08/transformer-novel-neural-network.html) e supera a performance dos métodos anteriores, baseados em RNNs e CNNs. A Transformer modela, de uma forma direta, os relacionamentos entre todas as palavras de um sentença, independente do posicionamento de cada uma delas.

Para computar a representação de uma dada palavra em uma sequência, a Transformer compara esta palavra com cada uma das outras presentes na sentença. Como resultado, obtém-se um score para cada uma das palavras já computadas, sendo que esta pontuação determina o quanto cada termo anterior contribui para a representação do próximo.

{:.image}
![A Transformer possui um stack de 6 camadas idênticas. No encoder, cada camada possui duas sub-camadas. A primeira implementa uma variação do mecanismo self-attention, chamado Multi-Head self-attention. A segunda é um simples rede feed-foward. O decoder também possui duas sub-camadas, mas ainda insere uma terceira, a qual aplica o método attention em toda a saída do encoder.](https://cdn-images-1.medium.com/max/2000/0*I2ng-cUnE3Rdm9O8.png)
*A Transformer possui um stack de 6 camadas idênticas. No encoder, cada camada possui duas sub-camadas. A primeira implementa uma variação do mecanismo self-attention, chamado Multi-Head self-attention. A segunda é um simples rede feed-foward. O decoder também possui duas sub-camadas, mas ainda insere uma terceira, a qual aplica o método attention em toda a saída do encoder.*

Mais detalhes desta abordagem podem ser obtidos por meio da leitura do paper “[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)” **(Vaswani et al. 2017).** Os autores também [disponibilizaram o código-fonte](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) em tensorflow, usado nos experimentos. Você pode tentar reproduzir os resultados que eles obtiveram. Existe uma implementação deste mecanismo usando o framework Pytorch e que é bem mais fácil de interpretar, [veja neste link](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

## Para concluir

Durante praticamente todo o post eu citei o exemplo da tradução neural de textos, mas você também viu que Seq2Seq pode ser aplicado em outros problemas de NLP (Natural Language Processing). Foi dito que isto também pode ser aplicado em Chatbots. Isto é interessante, porque no lugar de termos respostas previamente selecionadas para situações específicas, como ocorre nos CB baseados em regras, estas respostas seria exibidas completamente com base em critérios probabilísticos. Estas respostas seriam escolhidas pelo próprio chatbot para responder questões específicas. É algo que o deixaria mais autônomo e inteligente.

A criação de bases de conhecimento é também uma aplicação interessante. Imagine uma I.A que indexa o conteúdo de um blog como este e automaticamente cria uma seção FAQ com perguntas e respostas completamente baseadas no conteúdo encontrado? Isto já está sendo feito em larga escala, com o uso de seq2seq.

Agora, me fala uma coisa aí nos comentários! O que foi que você achou deste post? Te ajudou a entender um pouco sobre a teoria seq2seq, que você precisa para criar suas aplicações baseadas em NLP e I.A? Comenta aí!

## Referências, para um estudo mais aprofundado

[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[Transformer: A Novel Neural Network Architecture for Language Understanding](https://research.googleblog.com/2017/08/transformer-novel-neural-network.html)

[Natural Language Inference, Reading Comprehension and Deep Learning](https://nlp.stanford.edu/manning/talks/SIGIR2016-Deep-Learning-NLI.pdf)

[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), 2016

[A Neural Conversational Model](https://arxiv.org/pdf/1506.05869v1.pdf)
