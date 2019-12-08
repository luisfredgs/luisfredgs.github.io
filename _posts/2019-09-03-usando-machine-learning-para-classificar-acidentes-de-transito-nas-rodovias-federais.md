---
layout: post
title:  "Usando Machine Learning para classificar acidentes de trânsito nas rodovias federais do Brasil"
resume: "Recentemente eu usei machine learning para criar um classificador de acidentes ocorridos em rodovias federais do Brasil, de acordo com o nível de gravidade. Também fiz uma análise descritiva detalhada das ocorrências e estou compartilhando os resultados neste post."
date:   2019-09-01 23:50:00
categories: [data-science, machine-learning]
tags: [data-science, machine-learning]
permalink: "/machine-learning/classificando-acidentes-de-transito"
thumbnail: "/assets/img/usando-machine-learning-para-classificar-acidentes-de-transito.jpg"
---
{:.image}
![](/assets/img/usando-machine-learning-para-classificar-acidentes-de-transito.jpg)

{:.intro}
*Recentemente eu fiz uma análise descritiva nos dados de ocorrências sobre acidentes ocorridos em rodovias federais do Brasil, usando machine learning para criar um classificador de acidentes de acordo com o nível de gravidade.*


O sistema BR-Brasil, desenvolvido pelo Departamento de Polícia Rodoviária Federal (DPRF), cataloga todas os boletins de ocorrência registrados após algum acidente ter ocorrido em alguma rodovia federal. A ferramenta consolida diversos detalhes sobre os veículos e pessoas envolvidos, sobre as condições do local, a dinâmica do acidente, etc. Por meio do programa governamental de dados abertos, a PRF [disponibiliza publicamente](https://www.prf.gov.br/portal/dados-abertos) esses dados em formato legível, para que qualquer pessoa possa analisá-los.

Sabendo disso, resolvi pegar uma amostra desses dados, referente aos anos de 2017 e 2018, para fazer uma análise. O trabalho foi dividido em duas etapas. Na primeira, os dados passaram por um preprocessamento e algumas transformações. Em seguida, foi realizada uma análise exploratória e descritiva, para descoberta de conhecimento nesta base. Já na segunda etapa, foi desenvolvido um modelo para classificar os acidentes de acordo o nível de gravidade (leve, grave, gravíssimo). O modelo foi testado numa amostra com dados de 2019, com registros que vão de janeiro à maio e que não foram usados no conjunto de aprendizagem.

# Quais perguntas a análise ajudou a responder?

Para obter algumas respostas latentes nos dados, realizei uma análise descritiva e testes estatísticos de hipótese. Várias perguntas foram respondidas e abaixo estão listadas uma parte delas:

* As ocorrências acontecem mais frequentemente em qual BR?
* Existem valores extremos (outliers)? Onde eles estão?
* Acidentes envolvendo colisão traseira estão relacionados com o fato de ter caído chuva no momento do acidente?
* Quais são os estados com mais ocorrências
* Qual BR é mais perigosa de se trafegar no estado com mais ocorrências?
* Quais os tipos de acidentes que ocorrem mais?
* Em quais tipos de pistas acontecem mais acidentes (simples, dupla ou múltipla)?
* Em qual época do ano acontecem mais acidentes?
* Quais as maiores causas de acidentes?
* Em qual época do ano aconteceu mais acidentes provocados por ingestão de bebidas alcoólicas?
* Os acidentes acontecem mais nas retas ou nas curvas?
* Colisões e capotamentos acontecem mais nas curvas ou nas retas?
* O reparo de vias com o objetivo de reduzir acidentes é prioridade de quais estados?
* Onde se concentram mais os acidentes causados por animais na pista?
* Quais estados registram mais acidentes durante a noite?
* Quais as principais causas dos acidentes envolvendo vítimas fatais e onde eles mais acontecem?

# Preview do código em Python (+ links)

<div class="video-container">
	<center>
		<iframe width="560" height="315" src="https://www.youtube.com/embed/IOGdvLL2YTE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	</center>
</div>

O vídeo acima é apenas um preview do código desenvolvido em um notebook na plataforma Kaggle. Todos os detalhes sobre a análise desses dados, bem como os códigos em Python utilizados, podem ser encontrados [neste link](https://www.kaggle.com/luisfredgs/prf-an-lise-explorat-ria-classificador). Não se esqueça de dá um "up vote" no notebook para incentivar a produção de novos materiais similares.

Comenta aí em baixo, se você encontrou qualquer erro na metodologia ou no código, ou se ficou com alguma dúvida. Estou aqui para ajudar!

## Leia também

1. [7 livros essenciais para aprender machine learning]({% post_url 2018-10-22-7-livros-essenciais-para-aprender-machine-learning %})

2. [Dicas para aprender Machine Learning]({% post_url 2018-01-18-dicas-para-aprender-machine-learning %})

3. [Reconhecimento de Entidades Nomeadas (NER) — O que é? Quais são as aplicações?]({% post_url 2017-08-22-reconhecimento-de-entidades-nomeadas-ner-e-aplicacoes %})

4. [Classificando textos com Machine Learning]({% post_url 2018-05-07-classificando-textos-com-machine-learning %})

5. [Análise de sentimentos com redes neurais recorrentes LSTM]({% post_url 2018-12-23-analise-de-sentimentos-com-redes-neurais-recorrentes-lstm %})
