---
layout: post
title:  "Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?"
resume: "Nos últimos dois anos foram gerados cerca de 90% dos dados de que o mundo dispõe atualmente. Em torno de 80% destes mesmos dados não estão disponíveis de uma forma estruturada, sendo informações vindas de e-mails, posts de blog, redes sociais, textos gerados em sistemas de suporte ao cliente, etc."
date:   2018-12-24 11:50:00
categories: [nlp, deep-learning]
tags: [nlp, deep-learning]
permalink: /deep-learning/reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes
status: 1
---

{:.image}
![](/assets/img/reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes.png)

Nos últimos dois anos foram gerados cerca de 90% dos dados de que o mundo dispõe atualmente. Em torno de 80% destes mesmos dados não estão disponíveis de uma forma estruturada, sendo informações vindas de e-mails, posts de blog, redes sociais, textos gerados em sistemas de suporte ao cliente, etc. A chave para minerar e extrair valor desta montanha de dados é o Processamento de Linguagem Natural (PLN). O Reconhecimento de Entidades Nomeadas — *Named Entity recognition (NER)*, é uma das mais importantes ferramentas do Processamento de Linguagem Natural e refere-se à tarefa de extração de informação que é responsável por capturar as entidades presentes em um texto e classificá-las em categorias pré-definidas, tais como PESSOAS, EMPRESAS, LOCAIS, VALORES MONETÁRIOS, PORCENTAGENS e DATAS, que são as mais comuns, embora possa haver mais, dependendo do domínio da tarefa.

Uma metodologia típica do processo de reconhecimento de entidades nomeadas, doravante denominado **NER**, envolve a *aprendizagem supervisionada* de máquina (o modelo é gerado quando a máquina é exposta a um conjunto de dados de exemplo com classes previamente rotuladas), ou *semi-supervisionada* (apenas parte dos dados de treino possui classes rotuladas). Depois de gerar um modelo a partir dos dados de exemplo, o software consegue fazer inferências a partir de textos inéditos (reconhecer entidades), obtendo generalizações com base nos padrões aprendidos durante a fase de treino. É o que possibilita o NER sem grandes dificuldades, desde que você tenha dados de exemplo previamente disponíveis para o aprendizado da máquina, quanto mais dados disponíveis para o treino, melhor a precisão do modelo ao fazer as inferências.

Esta extração de informação normalmente ocorre a partir de dados não estruturados, como é o caso de textos de notícias, artigos de blogs, comentários de rede social, Tweets, ou reviews de produtos em sites de e-commerce. Trata-se de um recurso para o qual há inúmeras aplicações práticas. Veja algumas:

* Encontrar o nome de uma pessoa ou empresa em um tweet;

* Saber quais pessoas ou empresas foram mencionadas em um artigo de blog, texto de uma notícia, ou comentário em rede social (às vezes, o interesse é analisar as reações das pessoas e da mídia à uma determinada campanha de marketing);

* Quais produtos de uma empresa foram mencionados em uma reclamação no sistema de suporte ou numa rede social;

* Mapeamento e combinação de registros sobre uma mesma entidade a partir de várias fontes de dados (Resolução de Entidades). — Como exemplo, considere o caso em que as informações de saúde sobre um mesmo indivíduo possam está espalhadas em diferentes prontuários, em diferentes unidades de um sistema de saúde. Por meio da resolução de entidades, da qual o NER é uma subtarefa, estes dados podem identificados como pertencentes a uma mesma pessoa, em seguida mapeados e agrupados em uma única base, gerando informações úteis sobre a saúde de uma população inteira, ao longo de suas gerações.

* *Chatbots* — aplicações como reconhecer locais de destino em um chatbot de venda de passagens on-line;

* Softwares que respondem a perguntas diretas dos usuários, como é o caso da assistente virtual [POLIANA]({% post_url 2018-12-24-uma-assistente-virtual-na-raspberry-pi-usando-deep-learning %}), também fazem uso deste princípio.

Uma típica tarefa de NER é normalmente subdividida em duas subtarefas, que ocorrem praticamente ao mesmo tempo:

## Identificação das entidades

Identifica corretamente todas as entidades presentes no texto não estruturado, sem fazer a classificação de cada uma. Normalmente estas entidades passam ser representadas na forma de *tokens* — as palavras ficam armazenada em vetores:

{% highlight python %}
entities = ['ENTITIE-A', 'ENTITIE-B', 'ENTITIE-c']
{% endhighlight %}

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*ib5bGoWeBvF3Ju2V9HA-vg.png)

## Classificação das entidades identificadas

Classifica corretamente as entidades identificadas e armazenadas nos* tokens,*na etapa anterior.

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*lCbg3dIKIpKejzFmcCqQrA.png)

O software destaca e armazena as entidades que foram identificadas e classificadas no texto, de forma que os analistas, ou cientistas de dados possam tomar decisões com base nelas, ou o próprio software utiliza estas entidades como parte de um determinado procedimento (há um exemplo mais adiante).

Este tipo de recurso tem sido muito utilizado para melhorar o desempenho de diversas ferramentas como buscadores, sistemas comparadores de preços (utiliza resolução de entidades para mapear um mesmo produto em diferentes sites de e-commerce), softwares para mineração de opinião (análise de comentários no Twitter, mencionando entidades como um produto, ou uma empresa) e até mesmo chatbots que processam vendas com base em requisições escritas em linguagem natural.

## Aplicação em motores de busca

Uma vez que as frases utilizadas nos campos de busca do Google e do Bing são normalmente curtas, não estruturadas e ambíguas, surge a necessidade de uma metodologia capaz de analisar o curto texto semanticamente, detectar possíveis entidades presentes na frase e fazer a desambiguação de termos, permitindo que o usuário obtenha a melhor experiência possível nos resultados da busca orgânica. Google e Microsoft, de fato, aplicam NER com este objetivo. Se o algoritmo do Google consegue identificar uma determinada entidade presente nos termos de busca, ele irá, então, exibir uma coluna com mais informações sobre esta entidade, mapeadas a partir de diferentes fontes (Resolução de entidades), como no exemplo abaixo:

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*3HN_XLK6MzjkbwbmXWrzAw.png)

## Exemplo de aplicação de NER em Chatbots comerciais

Considere o seguinte exemplo:

Um usuário acessa a fanpage do Facebook de uma empresa que vende passagens rodoviárias, abre o chat e escreve:

> Eu quero uma passagem de Fortaleza para Crato, saindo no próximo domingo, às 22:00.

A esta altura você já deve saber que o Facebook Messenger possui uma API aberta, que permite a criação de *chatbots *integrados a sistemas de terceiros. Neste exemplo, o chatbot no Facebook Messenger processa o texto escrito pelo usuário e gera uma venda no sistema interno da empresa:

> Eu quero uma <Produto= **passagem**> de <Local= **Fortaleza>** para <Local= **Crato>**, saindo no próximo **<**Data= **domingo, às 22:00>**.

Ao longo do fluxo da conversa, várias informações importantes para a conclusão do processo de venda podem ser obtidas, as entidades importantes extraídas e, no final, dando origem à seguinte informação estruturada no sistema interno da empresa:

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*UCZjvumbXPR3KnIhjuhDLA.png)

## NER também pode ser aplicado em textos de redes sociais

Uma aplicação frequente de NER é a identificação e análise de reclamações postadas no Twitter, a fim de extrair nomes de empresas e produtos mencionados ao longo dos 140 caracteres escritos pelos usuários. Combinada com outras técnicas como a análise de sentimentos, isto constitui-se numa grande ferramenta de suporte à decisões, como parte de um processo de melhoria de serviços e produtos em uma empresa.

Veja o exemplo abaixo, onde um usuário reclama no Twitter, a respeito de um serviço mal prestado. É possível identificar algumas entidades, coisa que um software de NER poderia fazer automaticamente e emitir alertas para um sistema interno, conforme o caso. Um outro modelo de classificação poderia agir de maneira combinada, organizando estes tweets em uma fila (ou gerar tickets de suporte), de acordo com o nível de prioridade automaticamente atribuído a cada um (URGENTE, MÉDIA PRIORIDADE, BAIXA PRIORIDADE).

{:.image}
![](https://cdn-images-1.medium.com/max/2000/1*LcdllcOREIE7ezVjlvfSdA.png)

Diversas pesquisas acadêmicas no campo de NER têm sido conduzidas ao longo dos anos, gerando resultados excelentes (principalmente com o uso de redes neurais). Isto possibilitou o surgimento de uma grande variedade de aplicações comerciais baseadas em Processamento de Linguagem Natural, como produtos de empresas que investem pesados recursos em P&D. Estas funcionalidades hoje fazem parte de grandes ferramentas (geralmente na forma de APIs) como o [IBM Watson](https://www.ibm.com/watson/br-pt/), [Microsoft Azure Machine Learning](https://azure.microsoft.com/pt-br/services/machine-learning-studio/), ou [Google Cloud Platform — Cloud AI](https://cloud.google.com/products/machine-learning/?hl=pt-br).

Obviamente, há custos envolvidos no uso destas ferramentas, cujos valores são normalmente cobrados com base no volume de requisições mensais feitos às APIs (alguns cobram valores com base em horas). Dependendo da sua necessidade, estes custos podem ser facilmente diluídos, em virtude dos benefícios que podem ser obtidos. De todo modo, você sempre terá a opção criar a sua própria API, totalmente voltada para a sua necessidade — se você possui o conhecimento, ou equipe habilitada para tal — , ou investir na assinatura de serviços terceirizados. Esta última alternativa é bastante atraente, pois você não terá custos com manutenção de infraestrutura, ou de software. Só é preciso avaliar as opções.

## Veja também:

{:.image}
[![curso nlp](/assets/img/curso-processamento-de-linguagem-natural.png)](https://www.udemy.com/course/processamento-de-linguagem-natural-do-zero-a-producao/?referralCode=58A9E255EE0A65ACF871)

1. [Processamento de Linguagem Natural, do zero à produção](https://www.udemy.com/course/processamento-de-linguagem-natural-do-zero-a-producao/?referralCode=58A9E255EE0A65ACF871)

2. [Neural Architectures for Named Entity Recognition](http://neural%20architectures%20for%20named%20entity%20recognition/)

3. [Named Entity Recognition (LSTM + CRF) — Tensorflow](https://github.com/guillaumegenthial/sequence_tagging)

4. [How do I use IBM Watson to extract entity information from news articles?](https://www.oreilly.com/ideas/how-do-i-use-ibm-watson-to-extract-entity-information-from-news-articles)
