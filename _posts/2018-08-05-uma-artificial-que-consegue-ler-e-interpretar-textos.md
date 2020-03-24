---
layout: post
title:  "Machine Reading Comprehension — inteligência artificial que consegue ler e interpretar textos"
resume: "Afinal de contas, como é que uma máquina consegue ler e interpretar textos, do mesmo jeito que a gente faz? Você pode não ter percebido ainda, mas quando fazemos uma pergunta ao Google, ele executa um procedimento parecido com aquele que a gente fazia na escola. Explico."
date:   2018-08-05 11:50:00
categories: [nlp, deep-learning]
tags: [nlp, deep-learning]
permalink: /deep-learning/uma-inteligencia-artificial-que-consegue-ler-e-interpretar-textos
status: 1
---

Uma inteligência artificial capaz de compreender textos, do mesmo jeito que um ser humano faz, é algo bem mais tangível hoje do que foi em um passado não muito distante. Graças à evolução das técnicas de machine learning ao longo do tempo, hoje é possível ver este tipo de conceito aplicado em algumas ferramentas tecnológicas que usamos com frequência. 

Por exemplo, você pode nem ter percebido ainda, mas quando fazemos uma pergunta ao Google e ele vai procurar respostas em seu imenso banco de dados, um procedimento baseado em inteligência artificial que envolve interpretação de textos é executado instantaneamente. O Bing, da Microsoft, também usa este tipo de técnica. Até mesmo alguns chatbots estão usando isso. Na prática, é o que nos permite fazer perguntas de uma maneira muito mais natural ao Google e contirnuarmos, ainda assim, obtendo respostas relevantes pois o texto da query não precisa ser sintaticamente similar à alguma sequência contida no texto da resposta.

{:.image}
![](/assets/img/uma-inteligencia-artificial-que-consegue-ler-e-interpretar-textos-google.png)
*A aplicação de técnicas de machine learning para interpretação de textos permite que façamos perguntas para o Google de uma maneira muito mais natural, como se estivéssemos perguntando para um humano.*

Lembra daquelas atividades que tinham no seu livro de Português, História, Geografia? Havia um texto específico a ser lido e, nas páginas seguintes, estavam as perguntinhas sobre o texto que precisavam ser respondidas (a resposta sempre estava contida em alguma parte do texto)? Pois é, trata-se de uma tarefa trivial para qualquer humano. Para as máquinas, contudo, desempenhar essa mesma tarefa é algo bastante desafiador. Mas, então, como é que eles fazem para funcionar?

**Ao final deste post, você terá visto o seguinte:**

1. Como a inteligência artificial consegue interpretar textos;

1. Conjuntos de dados conhecidos;

1. Alguns algoritmos interessantes (Papers e códigos dos autores compartilhados no github para você tentar reproduzir os resultados);

1. Possíveis aplicações práticas (um curto vídeo de exemplo com chatbot funcionando no Facebook Messenger);

1. Fontes de estudo para leituras mais aprofundadas.

## Como uma inteligência artificial conseguiria tal façanha?

Depois de ler alguns papers sobre o assunto, me familiarizei com uma terminologia nova: **Machine Reading Comprehension (MRC)**. Apesar de a literatura normalmente mencionar desta forma, você pode também chamar de Compreensão de Leitura de Máquina (CLM). Estamos falando da habilidade de uma inteligência artificial fazer uso do deep learning para ler e compreender textos, tendo ainda a aptidão para responder perguntas baseadas em algumas passagens presentes no conteúdo.

Em boa parte da literatura, notei o uso de redes neurais artificiais com base nas arquiteturas **LSTM (Long-Short Term Memory)** ou **GRU (Gated Recurrent Unit),** normalmente bidirecionais, combinados com variantes específicas do mecanismo atencional de **[Bahdanau, et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)**. Nesse post, irei destacar alguns dos trabalhos que mais despertaram o meu interesse.

Por mais que você não esteja familiarizado com o tema, saiba que todos os dias você experimenta isto na prática enquanto faz pesquisas no Google ou no Bing, apenas para citar um exemplo. Um tipo de tarefa que tem sido muito explorada em modelos de MRC que contam com Deep Learning é o [Question Answering](https://en.wikipedia.org/wiki/Question_answering)– QA (Pergunta & Resposta). O pequeno vídeo abaixo ilustra bem este processo, que pode ter várias aplicações práticas, incluindo chatbots:

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/27As1HtqvJI" frameborder="0" allowfullscreen></iframe></center>
</div>

Muito embora não pareça, este tema não é nada novo. Por volta do ano 2000, isto já tinha relevância quando [Ng et al.](http://www.aclweb.org/anthology/W00-1316) desenvolveu um modelo de machine learning baseado em regressão logística para compreensão de leitura de máquina. Usado para responder perguntas aleatórias após ser exposto a qualquer texto inédito, o modelo foi o primeiro a mostrar que o uso de machine learning era capaz de alcançar resultados competitivos em MRC.

## O surgimento de Datasets mais precisos

Avanços mais significativos neste tópico passaram a ocorrer muito recentemente, principalmente com o uso mais abrangente das redes neurais artificiais e também devido à maior disponibilidade de dados para usar como treino. Não apenas isso, mas também devido à disponibilidade de datasets mais realistas e precisos. O resultado disto é que o MRC passou a ser um dos tópicos mais “quentes” na área de Processamento de Linguagem Natural (PLN).

[Rajpurkar et al. (2016)](https://arxiv.org/pdf/1606.05250.pdf) disponibilizou publicamente [o seu dataset](https://rajpurkar.github.io/SQuAD-explorer/) **S**tanford **Q**uestion **A**nswering **D**ataset (**SQuAD**), que viabilizou o desenvolvimento de diversos modelos interessantes. Trata-se de um dataset composto por **100.000+** perguntas elaboradas por humanos com base em um conjunto de artigos da Wikipédia. A resposta para cada pergunta é um segmento de texto destacado manualmente a partir de alguma passagem presente no artigo. Recentemente, foi libearada a versão 2.0 do dataset. O SQuAD2.0 testa a capacidade de um modelo não apenas responder às perguntas no processo de interpretação, mas também se abster de responder quando for feita uma pergunta que não pode ser respondida com base no conteúdo fornecido.

{:.image}
![Exemplo de pares de pergunta-resposta presentes no dataset SQuAD. Cada uma das respostas é um segmento de texto da passagem. Créditos: Rajpurkar et al. (2016)](https://cdn-images-1.medium.com/max/2000/1*RXS4n7ykoPKzG8S-Rlgfrg.png)*Exemplo de pares de pergunta-resposta presentes no dataset SQuAD. Cada uma das respostas é um segmento de texto da passagem. Créditos: Rajpurkar et al. (2016)*

Algum tempo depois, [Nguyen et al. (2016)](https://arxiv.org/pdf/1611.09268v2.pdf) disponibilizou o [dataset](http://www.msmarco.org/) **MS MARCO**: **MA**chine **R**eading **CO**mprehension Dataset. Refere-se à um conjunto de dados de alto padrão, cuidadosamente preparado com base em documentos coletados a partir do índice do buscador Bing. As respostas para as perguntas neste dataset também foram pontuadas por humanos, tornando este conjunto de dados tão preciso e real quanto o SQuAD, ou além.

# Alguns trabalhos interessantes

## BiDAF — Bi-Directional Attention Flow network

Diversos trabalhos interessantes surgiram diante da disponibilidade destes conjuntos de dados. [Seo et al., (2017)](https://arxiv.org/abs/1611.01603) propõe uma arquitetura hierárquica em seis camadas, a qual conta com um mecanismo antencional [Bahdanau, et al., 2015](https://arxiv.org/pdf/1409.0473.pdf) operando ao longo de um fluxo bidirecional. O modelo, que foi testado na solução de um problema de QA, utiliza uma variante do mecanismo atencional para obter uma representação do contexto referente à passagem textual, a qual está ligada a uma pergunta específica. Este fluxo ocorrendo em ambas as direções é justificado pelo ganho de informações que ele proporciona, ao obter dados contextuais em duas mãos.

No **BiDAF**, o contexto da passagem é observado e computado a partir de uma dada pergunta (Context2Query), e também no sentido contrário (Query2Context). A figura abaixo ilustra bem este processo:

![Uma ilustração do fluxo atencional bi-direcional. Créditos: **Seo et al., 2017**](https://cdn-images-1.medium.com/max/2000/1*r-LvP2My-IHymbzPa4r1hw.png)*Uma ilustração do fluxo atencional bi-direcional. Créditos: **Seo et al., 2017***

Você pode visualizar o modelo em funcionamento [por meio deste link](http://allgood.cs.washington.edu:1995/). Caso queira reproduzir os resultados, os autores disponibilizaram o [código no github](https://github.com/allenai/bi-att-flow).

## match-LSTM

Um outro algoritmo bem legal é o de [Wang & Jiang (2016a)](https://arxiv.org/pdf/1608.07905.pdf), o qual baseia-se no uso combinado de duas arquiteturas: **match-LSTM** e **Ptr-Net**. Funciona com base no seguinte princípio: Uma vez que é dada uma passagem de texto e uma pergunta relacionada com esta passagem, o objetivo é usar a referida combinação para identificar uma sequência dentro da passagem que corresponda à pergunta.

A **match-LSTM** percorre todos os tokens da passagem sequencialmente, computando um conjunto de pesos atencionais com o objetivo de medir o grau de relacionamento entre o *i-ésimo* token da passagem e o *j-ésimo* token da pergunta. Em seguida, estas duas partes são combinada em um vetor que é dado como entrada para uma LSTM unidirecional. A **match-LSTM** opera em duas direções com o objetivo de obter o contexto de cada token da passagem em ambos os sentidos da leitura (esquerda-direita/direita-esquerda).

A **Ptr-Net**, que representa a saída do modelo**,** fica encarregada de predizer onde começa e onde termina a sequência que corresponde à resposta para a pergunta. Deste modo, ela apenas precisa selecionar dois tokens a partir da passagem, a=(as,ae). Neste caso, as e ae representam o início e o fim da sequência que corresponde à resposta, respectivamente. Eles chamaram este método de **Boundary Model**. A figura abaixo ilustra esta arquitetura:

{:.image}
![a match-LSTM opera em duas direções, passando a saída para a camada Ptr, que tem o objetivo de predizer as fronteiras da sequência de resposta. Créditos: **Wang & Jiang (2016a)**](https://cdn-images-1.medium.com/max/2000/1*Xf5JNJV7HiZE3ix6GOed2Q.png)*a match-LSTM opera em duas direções, passando a saída para a camada Ptr, que tem o objetivo de predizer as fronteiras da sequência de resposta. Créditos: **Wang & Jiang (2016a)***

No mesmo trabalho, os autores também usaram uma outra variante do modelo, que não mencionei aqui, mas você pode conferir no paper que já foi linkado. Caso queira testar o código e tentar reproduzir os resultados, ele está disponível [neste link do github](https://github.com/shuohangwang/SeqMatchSeq).

## Reading Wikipedia to Answer Open-Domain Questions

Também despertou a minha atenção o trabalho de [Chen et al., (2017)](https://arxiv.org/pdf/1704.00051.pdf), o qual consiste num modelo que opera em larga escala. Ele é voltado para trazer respostas a perguntas em domínio aberto, usando a Wikipédia como única base de conhecimento. A arquitetura combina um componente de buscas (**Document Retriever**), baseado na medida TF-IDF, com uma rede neural LSTM de múltiplas camadas (**Document Reader**), a qual é treinada para encontrar respostas em múltiplos parágrafos de artigos da Wikipédia. Estes dois componentes trabalham da seguinte maneira:

1. **Document Retriever** — Um componente de buscas (não baseado em machine learning) é usado para retornar um subconjunto de artigos relevantes, com grandes chances de conter a resposta de uma pergunta. Para qualquer pergunta que é feita ao modelo, apenas 5 artigos da Wikipédia são retornados. A pergunta e os artigos obtidos são comparados usando a medida TF-IDF.;

1. **Document Reader** — Uma rede neural recorrente LSTM bidirecional recebe o retorno do primeiro componente do modelo e tenta identificar a resposta esperada a partir deste conteúdo. A rede neural agrega informação a partir de diferentes parágrafos para compor a resposta final.

{:.image}
![Uma ilustração do modelo QA de [**Chen et al., (2017)**](https://arxiv.org/pdf/1704.00051.pdf)](https://cdn-images-1.medium.com/max/2562/1*id4F3D0yOO74FNydnRqMrg.png) *Uma ilustração do modelo QA de [**Chen et al., (2017)**](https://arxiv.org/pdf/1704.00051.pdf)*

Para compor a base de conhecimentos usada para responder às perguntas de domínio aberto, os autores utilizaram um dump de todos os artigos da Wikipédia em inglês datado de 2016. Os autores também utilizaram o dataset SQuAD, para treinar o componente **Document Reader**.

> *"Nós estamos interessados em obter um sistema simples e completo, capaz de responder a qualquer pergunta usando a Wikipédia* " — [Chen et al., (2017)](https://arxiv.org/pdf/1704.00051.pdf)

Caso você esteja interessado no código para reproduzir os resultados do paper, os autores o [disponibilizaram no github](https://github.com/facebookresearch/DrQA).

## Possíveis aplicações

Como você já deve ter notado, modelos baseados em Machine Reading Comprehension têm mostrado uma incrível capacidade para entendimento da linguagem natural, considerando se tratar de uma tarefa que é bastante desafiadora até mesmo para uma inteligência artificial mais avançada. Este potencial abre portas para diversas aplicações práticas possíveis. Como mencionado no início do post, há aplicações em mecanismos de buscas e chatbots, mas são apenas dois exemplos e pode haver bem mais.

Você poderia, por exemplo, usar isto para desenvolver um chatbot e disponibilizá-lo na página de vendas de um produto (o cliente poderia perguntar: *Qual é a configuração de memória deste notebook?*) ou na página FAQ do seu site. Assistentes virtuais que funcionam em sistemas de EAD também podem se beneficiar deste tipo de tecnologia.

Para que você possa visualizar melhor, achei que seria interessante exemplificar por meio de um pequeno vídeo, ilustrando uma situação real. Digamos que seu chatbot seja responsável por tirar dúvidas dos visitantes em uma página de produto. Então, ele normalmente teria como base de conhecimento as especificações do produto, o que na literatura já vista neste post é normalmente referenciado como *passagem de texto*. Veja um exemplo abaixo:
> Informações sobre o produto podem ser obtidas aqui. O Moto G6 Plus possui sensor de impressão digital multifunção. Com ele, você não precisa mais digitar senha quando quiser acessar os aplicativos do seu smartphone. Basta usar o sensor de impressão digital multifunção que, além de desbloquear ou bloquear seu aparelho, pode ser utilizado como botão de navegação, para que a tela tenha um maior aproveitamento do espaço. O Moto G6 Plus tem Processador Qualcomm Snapdragon 630 Octa-Core 2,2 GHz com recursos gráficos avançados, para que você execute aplicativos e navegue na web. A bateria de 3200 mAh do Moto G6 Plus tem capacidade suficiente para o dia inteiro. Para isso, ela é turbinada pelo super carregador TurboPower, que fornece até 6 horas de uso com apenas 15 minutos de carregamento. Você pode pagar com cartões VISA e MASTER.

Neste caso, sempre que alguém fizer qualquer pergunta relacionada à passagem acima, seu chatbot conseguirá responder. Veja, no vídeo abaixo, o exemplo de um chatbot que aplica este princípio, funcionando no Facebook Messenger:

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/isI9UuuBcbs" frameborder="0" allowfullscreen></iframe></center>
</div>

Claro, ainda há muito espaço para melhorar. Toda semana, praticamente, alguém publica um paper com uma arquitetura nova que supera os resultados obtidos em trabalhos anteriores. É interessante ficar de olho e sempre tentar reproduzir os resultados depois de ler estes papers, compreender a arquitetura dos modelos e tentar rodar os códigos que quase sempre são disponibilizados pelos autores no github.

## Leitura aprofundada

Para uma leitura mais aprofundada, consulte os seguintes papers:

[Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf), 2017

[Machine Comprehension Using match-LSTM and Answer Pointer](https://arxiv.org/pdf/1608.07905.pdf), 2016

[Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf), 2016

[Neural Machine Translation By Jointly Learning To Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), 2014

[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf), 2016

[MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268v2.pdf), 2016
