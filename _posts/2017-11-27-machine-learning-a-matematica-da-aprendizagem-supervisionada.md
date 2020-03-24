---
layout: post
title:  "Machine Learning — a matemática da aprendizagem supervisionada"
resume: "O que torna possível o Machine Learning? Como é que uma máquina consegue aprender a partir de um conjunto de informações e ainda ser capaz de dá palpites sobre coisas que nunca viu?"
date:   2017-11-27 11:50:00
categories: [machine-learning]
tags: [machine-learning]
permalink: /machine-learning/machine-learning-a-matematica-da-aprendizagem-supervisionada
status: 1
---

<small>*Atualização mais recente: 23/06/2019*</small>

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/WgUrONLhons" frameborder="0" allowfullscreen></iframe></center>
</div>

{:.intro}
*O que torna possível o Machine Learning? Como é que uma máquina consegue aprender a partir de um conjunto de dados e ainda usar esse conhecimento para tomar decisões sobre coisas que ela nunca viu?*

O Machine Learning é uma realidade agora e muitas pessoas, gestores ou não, já tomaram consciência do papel que a inteligência artificial terá na sociedade e na sobrevivência de muitos negócios pelo mundo, a partir de agora. Colossos da tecnologia, tais como a Microsoft, Google, Facebook, IBM, Oracle e Nvidia têm investido crescentes somas de dinheiro em P&D com foco nesta área. É fácil notar que esses esforços estão tornando o uso da inteligência artificial cada vez mais acessível para nós. Exemplos são o desenvolvimento de ferramentas como o [Tensorflow](https://www.tensorflow.org/), ou o [Pytorch](https://pytorch.org/), que abstraem uma parte expressiva da complexidade de criar modelos de machine learning com redes neurais. Há, ainda, diversas APIs prontas que disponibilizam modelos pré-treinados para diversas finalidades e você só precisa chamar um endpoint a partir do seu aplicativo.

São iniciativas que tiveram início tempos atrás, preparando o terreno para o que estamos vivenciando hoje, na era da inteligência artificial. Carros que dirigem sozinhos; câmeras inteligentes que monitoram gôndolas em supermercados e alertam os gestores quando chega o momento ideal para repor os produtos; robôs que interpretam peças judiciais e agilizam o trabalho dos profissionais da área jurídica; robôs que analisam milhares de perfis de candidatos à vagas de emprego, cruzam diversas informações e encontram o melhor candidato para uma vaga específica, agilizando o trabalho dos profissionais de RH. Estamos falando de algo que está revolucionando mercados, tendo impulsionado a criação de diversos produtos inovadores e competitivos. Agora, te faço uma simples pergunta: Você já parou para pensar sobre o que torna tudo isso possível? A verdade é que tudo isso seria pouco tangível sem a ajuda da **MATEMÁTICA**!

O objetivo desse post é apresentar a você os princípios que governam o funcionamento de alguns algoritmos de machine learning amplamente usados na indústria, numa categoria de aprendizagem específica. Deixo as referências no final para que você possa se aprofundar.

# Quais são os princípios básicos do Machine Learning?

Machine Learning é um subcampo da inteligência artificial, que se vale de conceitos matemáticos amplos com o objetivo de fazer com que algoritmos "aprendam" a partir de dados. Trata-se de usar dados para encontrar e ajustar uma função ótima em um conjunto de parâmetros dentro de um grande espaço de funções possíveis. Uma vez encontrada esta função ótima, ela poderá ser utilizada para fazer extrapolações diante de situações variadas. Em outras palavras, uma função deste tipo poderá ser utilizada para fazer diversas análises preditivas. Exemplos são: predizer quando um equipamento irá falhar, quando conceder crédito a um cliente de banco, qual é o limite ideal para liberar no cartão de crédito dele, como encontrar a melhor rota de tráfego em dias de engarrafamento, identificar determinados objetos em cenas em tempo real ou identificar cenas de violência (útil em monitoramente de imagens urbanas), identificar automaticamente o assunto de um post no twitter, identificar conteúdos ofensivos pela web, etc....

> Machine learning não se trata de programar a máquina para executar uma sequência de instruções, mas de ensiná-la a partir de dados

De um modo geral, o ML é uma junção de **matemática** (álgebra linear, cálculo, estatística, teoria da probabilidade, teoria da informação, etc), **software** e **dados** (estruturados ou semi-estruturados). Assim, tal combinação permite que as máquinas “aprendam” e desempenhem tarefas específicas sem serem explicitamente programadas para isto. Portanto, não estamos falando daquela abordagem determinística tradicional, normalmente baseada em regras, onde a máquina segue um “roteiro” que orienta todos os passos a serem seguidos e cada resultado é previamente determinado. O ML, pelo contrário, atua em um ambiente onde a incerteza é um conceito chave. Ao ser exposto a uma certa quantidade de dados (dados de treino), o algoritmo de ML consegue identificar padrões e aprender com eles, gerando modelos capazes de obter generalizações (predição) em dados completamente novos. 

Atualmente, as quatro formas de aprendizado de máquina mais utilizadas são:

* **Supervisionada** — normalmente usado em tarefas de classificação, onde os dados de treino são totalmente rotulados (a saída desejada é conhecida) e orientam o aprendizado da máquina. É como se um professor estivesse presente o tempo inteiro durante a fase de aprendizado. É o método de aprendizagem mais utilizado hoje em dia;

* **Semi-supervisionada** — apenas parte dos dados possui rótulos e o restante não possui uma saída conhecida. Neste caso, o algoritmo baseia-se nos dados rotulados para formar agrupamentos daqueles que não foram rotulados, normalmente agrupando aquelas informações que possuem alguma similaridade entre elas;

* **Não supervisionada** — frequentemente utiliza-se este método de aprendizado em análises exploratórias, onde o algoritmo não conhece a saída dos dados e precisa identificar estruturas, regularidades e relacionamentos nos dados para, então, apresentar uma saída.

* **Aprendizagem por reforço** - Trata-se de otimizar a assertividade de um agente com base em feedbacks que ele recebe do meio onde está inserido. Você tem o agente, um espaço de estados e um conjunto de ações que ele precisa tomar. Cada ação tomada pelo agente promove uma mudança de estado (mover para frente, tira o agente de uma posição inicial) e o resultado pode ser o desejado ou não. Se o resultado por tomar uma ação específica for o desejado, o agente recebe uma recompensa, como um incentivo por ter agido corretamente. Do contrário, ele recebe uma penalidade (para desencorajá-lo a repetir uma ação que levou a um resultado ruim). O objetivo é encontrar o melhor conjunto de ações que permita alcançar um objetivo específico. Este método é muito utilizado na área da robótica e, recentemente, tem sido usado com redes neurais para treinar carros autônomos baseado no feedback de um conjunto de sensores em torno do veículo, que o ajudam a tomar decisões ótimas. Foi também o método que a DeepMind usou para treinar o seu famoso [algoritmo campeão](https://deepmind.com/research/alphago/) de *Go*.

A *aprendizagem supervisionada* é, de longe, a mais comum, onde a relação entre a entrada e a saída $f(x) = y$ dos dados de treino já é conhecida e o algoritmo não precisa tentar encontrar esta relação sozinho. Neste texto, nós vamos focar em aprendizagem supervisionada.

{:.image}
![](https://cdn-images-1.medium.com/max/2038/0*5kc8YLeJhDTqK73X.png)

Para que a *aprendizagem supervisionada* possa gerar modelos capazes de fazer boas predições, é preciso tomar alguns cuidados, tais como conhecer bem o problema a ser resolvido, ter dados representativos no domínio da tarefa e **escolher o algoritmo mais adequado** para cada situação. Para que isto seja possível, é necessário que você conheça estes algoritmos e saiba quando usar cada um deles (ou quando não usar). Mas, que algoritmos são esses? Já é o momento de apresentá-los:

## Qual é a matemática por trás da aprendizagem supervisionada?

### k-Nearest Neighbors

Frequentemente aplicado em mineração de dados, este é um dos mais básicos no leque de ferramentas de um Engenheiro de Machine Learning (e é também um algoritmo bem legal). O *k-Nearest-Neighbor* (*K* vizinhos mais próximos) é muito adotado em problemas de classificação e regressão. A predição em cima de novos dados é feita quando o *KNN* consegue encontrar os pontos de dados mais próximos a estes no conjunto de dados de exemplo.

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*qKsLIWj0t9wx7fzx.png)

Na ilustração acima, os pontos em formato circular representam os dados de exemplo (ou de treino), ao passo em que os pontos em formato de **x** representam dados ainda não “vistos” pelo modelo. Dado um $k$ inteiro positivo e uma observação $x$ (como mostrado na figura), o classificador *KNN* primeiro identifica os $k$ pontos no conjunto de treino que são mais próximos de $x$ (na figura, *k=1*). Em seguida, o modelo estima a probabilidade condicional para uma classe $j$ com base na fração dos pontos de dados de treino cujos valores de classe (o $y$ da função) são iguais a $j$. Para compor a saída, o modelo aplica a regra de Bayes e classifica a observação de teste *x* com a classe que possuir a maior probabilidade de ser a correta (considerando as classes dos pontos na vizinhança). Trazendo isto para dentro de um contexto mais matemático, o *KNN* funciona assim:

$$
  \text{Pr}(Y=j | X=x_0) = \frac{1}{K}\sum_{i \in \mathcal{N}_0}I(y_i=j)
$$

Como você já deve ter notado, o KNN não é daqueles algoritmos que você precisa treinar para obter um modelo, visto que basta ter um conjunto de dados armazenado e rodar o algoritmo. Considerando que o algoritmo vai classificar um ponto de consulta novo com base na vizinhança dele, é importante escolher com sabedoria a quantidade de vizinhos deste ponto a ser levada em conta durante o cálculo. Aliás, este um ponto extremamente importante, pois uma escolha ruim para este parâmetro pode fazer dele um algoritmo inútil. Além disso, como ele faz uso de métricas de distâncias, como a Euclidiana, os resultados que você vai obter estarão sob influência disso também. Uma estratégia comum é ponderar os pontos de dados da vizinhança com base no valor inverso das distâncias. Assim, os vizinhos que estão mais próximos do ponto de consulta têm mais influência no resultado do que aqueles que estão mais afastados.

O KNN tem a evidente desvantágem de consumir bastante recurso computacional quando você possui um conjunto de dados muito grande. O custo é proporcional ao tamanho da sua base.

### Regressão Linear

A regressão linear é uma outra abordagem que está entre as mais simples, sendo amplamente utilizada por cientistas de dados. Foi desenvolvida no campo da estatística e é tratada como um método útil para entender o relacionamento entre valores numéricos de entrada e saída. O ML a “tomou emprestado” e, assim, a regressão linear se tornou uma das ferramentas mais úteis neste campo. Torna-se uma abordagem útil, por exemplo, sempre que o seu objetivo é identificar relacionamentos entre variáveis e estimar o impacto que uma ou mais variáveis independentes causam em uma variável alvo. Por exemplo: Será que o número de quartos de uma casa, a latitude/longitude e o fato de ter ou não uma garagem influencia no nível de renda de uma pessoa? Qual a influência de cada uma destas variáveis na renda do indivíduo, caso exista alguma relação entre elas? Trata-se de um método largamente utilizado e é importante que você tenha um bom entendimento em torno dos princípios da regressão linear, considerando que algumas metodologias de aprendizado mais complexas costumam ser vistas como generalizações ou extensões dela.

Resumidamente, o objetivo é estimar uma variável quantitativa ***Y*** com base em preditor(res) ***X*** e assume-se que haja um relacionamento linear entre ***X*** e ***Y***. Matematicamente, nós podemos descrever a regressão linear assim:

$$
  Y \approx \beta_0 + \beta_1X
$$

$$ \beta_0 $$ e $$ \beta_1 $$ são coeficientes (também chamados de parâmetros) cujos valores não se conhece e serão estimados com base nos dados de treino.

Uma vez obtidos os valores aproximados destes parâmetros, seria possível, por exemplo, estimar um valor em vendas $$ Y $$ com base no quanto se está investindo em anúncios de Facebook $$ X $$. Isto ajuda a responder perguntas como: Qual plataforma de anúncios contribui mais para as vendas (qual variável de entrada afeta mais o valor da saída)? A regressão linear tem algumas suposições que não podem ser violadas, para que você possa obter um modelo que preste. Uma dessas suposições é a de que a variável dependente $$ Y $$ precisa ter uma distribuição normal. Caso a suposição de normalidade seja violada, o ideal é você partir para os [Modelos Lineares Generalizados](https://en.wikipedia.org/wiki/Generalized_linear_model). Um post interessante sobre regressão linear pode ser encontrado [aqui](https://machinelearningmastery.com/linear-regression-for-machine-learning/).

### Regressão Logística

A regressão logística é a análise de regressão utilizada quando a variável dependente é categórica, normalmente assumindo valores binários (como **0** ou **1**). Ao ser utilizada em problemas de classificação, ela pode estimar a probabilidade de ocorrência de uma classe específica com base nos valores das variáveis de entrada. Não tão diferente da *regressão linear*, a *regressão logística* torna-se um pouco diferente basicamente pelo uso da função que dá o nome a esta modelagem: *função logística*.

Considerando que o interesse aqui é obter a probabilidade de ocorrência de um determinado evento (classe), dadas as variáveis de entrada, os valores de saída deverão, assim, está contidos no intervalo [0,1]. Desta forma, para qualquer variável de entrada $$ X $$, a função logística garantirá que a saída seja um valor compreendido entre 0 e 1. Isto acontece porque a função logística satura diante de valores abaixo de **0** e acima de **1**. Veja:

$$
  p(x) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X} }
$$

A *função logística* é representada pela seguinte curva:

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*i1Fn-E-p5nFqL1aJ.png)

Eu recomendo o livro de *James, Gareth (et al.)*, que traz uma ótima explicação sobre este assunto, e pode ser encontrado [neste link](http://amzn.to/2zsQQmk).

### Support Vector Machines (SVMs)

O *SVM (Máquina de Vetores de Suporte)* funciona tanto para regressão quanto para classificação. No caso da classificação, o algoritmo toma como entrada um conjunto de observações e estima, para cada valor dado, a classe à qual pertence este valor. O SVM consegue obter um *hiperplano* capaz de separar os dados em classes distintas. Pode ser usado tanto para classificar dados que sejam linearmente separáveis quanto aqueles que não são linearmente separáveis por um *hiperplano*. É essencialmente um classificador binário e linear, mas pode ser estendido para classificar dados em mais de duas classes e que não possam ser separados linearmente. Para isto, o algoritmo conta com uma técnica chamada *Kernel Trick*, a qual constitui-se numa ponte entre a linearidade e a não-linearidade*. Um *kernel* é uma função que quantifica a similaridade entre duas observações em um espaço de maior dimensionalidade. 

Exemplo de um *kernel* polinomial não linear, quando *d > 1*:

$$
K(x_i,x_{i'}) = (1 + \sum_{j=1}^px_{ij}x_{i'j})^d
$$

A equação acima representa um tipo de kernel utilizado no SVM (não é o único), sendo conhecido como *kernel polinomial* de grau *d*, onde *d* é um número inteiro e positivo. Quando *d = 1*, a equação torna-se linear. Há outras funções de kernel que podem ser utilizadas, como a Radial Basis Function - RBF, que é amplamente utilizada. Mas também você pode fornecer suas próprias matrizes de kernel em situações específicas. Há algum tempo eu gravei um vídeo explicando o algoritmo SVM em detalhes (tem código em Python). Veja só:

<div class="video-container">
	<center>
		<iframe width="560" height="315" src="https://www.youtube.com/embed/b8nP9g0p8X4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	</center>
</div>

### Árvores de decisão e *Random Forests*

Estão entre os modelos de ML mais populares. As árvores de decisão dividem todo o espaço de decisão em subregiões, com o objetivo de simplificar o processo de predição. São algoritmos bastante úteis e de fácil interpretação. Uma abordagem bem comum é produzir multiplas árvores e combinar suas predições, visando obter um único modelo que consolida as decisões de vários modelos obtidos individualmente - *costumamos referenciar isto como um Ensemble*. Isso geralmente resulta em uma melhoria expressiva na acurácia do modelo final e é uma técnica bastante utilizada em diversas situações práticas. Apenas para citar um exemplo, o sensor de movimentos do [Kinect ](https://pt.wikipedia.org/wiki/Kinect)(o famoso acessório do console Xbox, da Microsoft), utiliza um modelo classificador baseado em um [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) de árvores de decisão. Trata-se de um conjunto de modelos em árvore chamado *Random Forests.* Lembre-se deste detalhe quanto estiver jogando *Just Dance* no Xbox One ;-).

{:.image}
![](https://cdn-images-1.medium.com/max/2048/0*afZyTZPv8WVw3rbq.jpg)

No contexto destes modelos baseados em árvore, o *Random Forests* é um dos mais populares. Trata-se de um **conjunto** de múltiplas árvores de decisão que operam em subdivisões do espaço de entrada, onde uma quantidade aleatória de árvores é treinada em partes específica do conjunto de aprendizagem. As predições destas árvores individuais são, então, combindas pela média para obter um modelo final que é significativamente mais robusto do que o melhor modelo obtido individualmente. Quando eu digo *combinadas pela média*, quero dizer que acontece o seguinte:

$$
\hat{f}_{\rm avg}(x) = \frac{1}{B}\sum_{b=1}^B\hat{f^b}(x)
$$

Ou seja, nós calculamos várias predições $\hat{f^1},\hat{f^2},\hat{f^3},...,\hat{f^B}$ individualmente, em $B$ partes diferentes do conjunto de aprendizagem e depois juntamos essas predições pela média. Trata-se de um princípio que já é usado em um outro método chamado [*bagging*](https://en.wikipedia.org/wiki/Bootstrap_aggregating), a partir do qual o Random Forest é extendido.

Ao ser usado em tarefas de classificação, por exemplo, o *random forest* vai considerar a estimativa (o voto) de cada árvore do conjunto para obter a classificação com base na classe que foi mais votada. O parâmetro mais importante a ser escolhido durante a modelagem com Random Forest é o número de árvores de decisão usado pelo Ensemble. Deve ser escolhido com parcimônia, pois se o número de árvores for grande, seu modelo entrará em overfitting rapidamente.

A figura abaixo ilustra bem o funcionamento deste algoritmo. 

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*p2xObVSbK8WBxb77.png)

Da próxima vez em que você ouvir alguém falar a frase “várias cabeças pensam melhor do que uma”, lembre de que o *Random Forest* baseia-se exatamente neste princípio.

### Redes Neurais Artificiais

As redes neurais artificiais são algoritmos de Machine Learning que se baseiam no funcionamento do cérebro biológico e suas conexões sinápticas. Aliás, é por conta delas que cunhou-se o termo [*deep learning*](https://pt.wikipedia.org/wiki/Aprendizagem_profunda) (aprendizagem profunda), o qual envolve o uso de redes neurais com várias camadas. Por muitos anos, o uso de RNAs se manteve inviável, dado que sempre exigiram tempo e capacidade computacional consideráveis. Mas este panorama mudou a partir do momento em que a [computação paralela](https://pt.wikipedia.org/wiki/Computa%C3%A7%C3%A3o_paralela) ganhou um novo impulso, graças ao surgimento de GPUs (Unidades de Processamento Gráfico) cada vez mais poderosas, ao mesmo tempo em que o custo de aquisição destes hardwares se tornou bem mais acessível. De fato, o poder de processamento paralelo oferecido pelas GPUs melhora significativamente a performance de treino de uma rede neural, reduzindo o tempo de ajuste do modelo de maneira expressiva. Por tal motivo, estes dispositivos se tornaram essenciais para os processos de criação de inteligências artificiais com *deep learning*. Atualmente já existem serviços de Cloud Computing com GPUs dedicadas e especialmente dimensionados para tarefas de *aprendizagem profunda*, como o [GPU Cloud Computing](https://www.nvidia.com/en-us/data-center/gpu-cloud-computing/), da Nvidia, o [Amazon EC2 Elastic GPUs](https://aws.amazon.com/pt/ec2/elastic-gpus/), [Microsoft (Azure)](https://docs.microsoft.com/pt-br/azure/virtual-machines/windows/sizes-gpu#nd-series) e [Google (Google Cloud)](https://cloud.google.com/deep-learning-vm/?hl=pt-br).

As RNA’s tornam-se bastante úteis a partir do momento em que passa a ser difícil resolver problemas com as abordagens mais tradicionais de aprendizado de máquina. Elas podem ser empregadas em tarefas como:

* Reconhecimento facial

* Detecção de objetos em cenas

* Análise de imagens médicas

* Análise de séries temporais

* Processamento de sinais

* etc

Uma RNA é composta por neurônios artificiais interligados e organizados em camadas. Cada neurônio artificial é representado por uma função matemática que computa os pesos sinápticos (parâmetros) da rede. O ajuste destes parâmetros é feito automaticamente e em diversas iterações e determina o aprendizado da rede. Uma rede neural clássica consiste de basicamente três camadas: **1)** Camada de entrada, **2)** camada intermediária (também chamada de camada oculta) e **3)** camada de saída. Todas elas são interligadas por funções de ativação não lineares.

{:.image}
![](https://cdn-images-1.medium.com/max/2048/0*TNp6H_4irvEyxUMu.jpg)

A figura acima ilustra um modelo de neurônio artificial chamado ***Perceptron*** (lado direito), que é a base para as redes neurais multicamadas, comparando-o com um neurônio biológico (lado esquerdo). Os dendritos, que correspondem aos dados de entrada $$ \{x_1, x_2, ..., x_m\} $$, também chamados de *features*, estão associados ao neurônio artificial por meio dos pesos sinápticos $$ \{w_1, w_2,.., w_m\} $$. Opcionalmente, um neurônio Perceptron pode ter uma entrada extra como sendo uma constante chamada *bias*, cujo valor (também chamado de valor de ativação) é sempre 1 e cujo peso associado é denotado por *w0*. O núcleo do corpo celular é representado por uma função somatória que combina as entradas do neurônio (eu prefiro chamar de combinador linear). 

Uma função de ativação, por sua vez, é aplicada ao resultado do somatório, gerando um valor de saída no axônio. A ideia central é extrair combinações lineares das entradas e, então, modelar a saída como uma função não linear. Para isto, costuma-se utilizar uma função de ativação não-linear, como a [*ReLU*](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

Há uma grande variedade de arquiteturas de RNA disponíveis atualmente, com alguma arquitetura nova surgindo praticamente toda semana. A mais comum e básica RNA multicamadas é um agrupamento de vários *Perceptrons* interligados, também chamado de *Perceptron Multicamadas*. Durante o treino, o algoritmo recebe um conjunto de dados como entrada $$ x_n $$, ajusta os pesos $$ w_m $$ e produz uma saída $$ y_k $$. Entretanto, normalmente a saída é acompanhada de um erro, que seria a diferença entre o valor alvo (valor de saída desejado) e a saída que foi produzida pela rede. 

O objetivo é promover tantas iterações quanto possível nos dados de treino até que este erro seja minimizado por meio de um processo de otimização de uma função que, preferencialmente, deveria ser convexa. Durante estas iterações, os pesos serão ajustados inúmeras vezes até que a rede seja capaz de generalizar bem. Estes ajustes nos pesos ocorrem por meio da [*retropropagação do erro*](http://neuralnetworksanddeeplearning.com/chap2.html) produzido na saída, na qual a rede o propaga pelo caminho inverso do fluxo (camada por camada), utilizando derivadas parciais para computar o gradiente da função objetivo, tendo como base o valor deste erro para promover o ajuste dos pesos até que tal erro se reduza ao mínimo possível. Usando um conjunto de derivadas parciais de primeira ordem, computadas pela regra da cadeia do cálculo, com valores ponderados por uma taxa de aprendizado, o gradiente vai indicar a proporção do ajuste que precisa ser feito nos pesos da rede a cada iteração.

O diagrama abaixo ilustra uma rede neural MLP, também chamada de *feed-foward* (alimentada adiante). Ela tem duas camadas, sendo que uma delas é oculta.

{:.image}
![](https://cdn-images-1.medium.com/max/2000/0*1WQ3MbOKn2GXSO-b.png)

Perceba que os *pesos* (parâmetros) são representados por ligações entre cada *nó* (neurônio) da rede. As setas verdes indicam o fluxo no qual ocorre o aprendizado. A seguinte função ilustra estes estágios:

$$
y_k(\mathbf{\text{x}}, \mathbf{\text{y}}) = \sigma \bigg( \sum_{j=1}^Mw^{(2)}_{kj}h\bigg(\sum_{i=1}^Dw_{ji}^{(1)x_i} + w_{j0}^{(1)}\bigg) + w_{k0}^{(2)} \bigg)
$$

Na equação acima,  $\sigma$ é uma função de ativação (como uma ReLU ou Tangente hiperbólica), a qual recebe os valores das entradas $x_i$ ponderadas pelos pesos $w$, incluindo os das camadas intermediárias. O processo de aprendizagem da rede neural consiste justamente em obter a melhor combinação possível destes pesos, de modo que a rede produza o menor erro e generalize o melhor possível. É evidente, contudo, que esta é uma representação simplificada do processo de aprendizado de uma RNA, visto que o interesse aqui **não** é abordar este assunto de uma forma aprofundada. Por este motivo, eu recomendo que você veja o [livro do Simon Haykin](http://amzn.to/2Bs5T0P), o qual traz um estudo bastante abrangente das RNA’s.

Como mencionado, há diferentes arquiteturas de redes neurais, com propósitos específicos. A maioria delas deriva de redes neurais convolucionais (CNN) e redes recorrentes (RNN):

**Redes Neurais Recorrentes** — São muito utilizadas em tarefas relacionadas com o processamento de linguagem natural, como modelagem e marcação de sequências, reconhecimento de entidades, análise de sentimentos, reconhecimento de voz, tradução de textos, etc. Caso você queira ter uma visão mais ampla desta arquiteura neural, bem como para ver um exemplo de aplicação na prática com códigos em Python, sugiro que veja este vídeo:

<div class="video-container">
	<center>
		<iframe width="560" height="315" src="https://www.youtube.com/embed/bIcadBu--u8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	</center>
</div>

**Redes Neurais Convolucionais** — São muito utilizadas no campo de visão computacional, como classificação de imagens e reconhecimento de objetos em fotografias e vídeos, além de outras aplicações mais recentes como as GANs - Generative Adversarial Networks. As GANs permitiram o desenvolvimento de algumas aplicações interessantes, como é o caso de modelos capazes de [gerar fotografias de pessoas que nunca existiram](https://thispersondoesnotexist.com/). Recentemente, as ConvNets revelaram uma grande capacidade para processar linguagem natural, trazendo resultados bastante interessantes. Em algumas situações, elas são combinadas com RNNs e outras técnicas, possibilitando outros resultados bacanas.

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/FhwzOaEMk6Y" frameborder="0" allowfullscreen></iframe></center>
</div>

As RNA’s entram em cena para nos ajudar a resolver problemas que normalmente parecem ser intratáveis. Problemas estes que envolvem não apenas uma complexidade maior, como também volumes de dados consideravelmente grandes. São algoritmos que superam consideravelmente as performances de abordagens mais tradicionais (mas nem sempre!!!!!)

Atualmente, grandes produtos e serviços que fazem parte do nosso cotidiano e facilitam a nossa vida contam com tecnologias neurais. O Google Tradutor, por exemplo, já está na sua segunda geração, [agora contando com modelos neurais](https://www.blog.google/products/translate/higher-quality-neural-translations-bunch-more-languages/) capazes de compreender o contexto das sentenças e proporcionar traduções muito melhores. A Apple, por sua vez, está utilizando tecnologias baseadas em deep learning para tornar os seus produtos muito mais competitivos, como é o caso das versões mais recentes do IOS e do [HomePod](https://www.apple.com/homepod/). Os iPhones mais recentes, por exemplo, [conta com um chip neural](https://www.tecmundo.com.br/produto/122085-apple-comecou-desenvolver-a11-bionic-iphone-x-tres-anos.htm).

### Para concluir

O objetivo deste post foi trazer uma visão geral, **não aprofundada**, dos algoritmos de aprendizagem supervisionada mais frequentes em tarefas de Machine e Deep Learning. Embora eu tenha tentado limitar a apresentação de jargões matemáticos, é preciso entender que o Machine Learning é, por natureza, fundamentado nesta ciência. Você não precisaria se preocupar muito com isso caso o seu interesse seja apenas incorporar alguma solução existente de ML dentro do seu negócio. Entretanto, caso pretenda criar seus próprios modelos ou outras soluções baseadas em IA com o apoio de modelos customizados, saiba que terá que estudar ML com mais carinho :-). 

Ficou interessando no assunto? Para saber como iniciar seus estudos em Machine Learning, não deixa de ver [este outro post, com dicas de como começar](https://luisfred.com.br/machine-learning/dicas-para-aprender-machine-learning). Também [criei um grupo de estudos recentemente](https://medium.com/luisfredgs/grupo-de-estudos-de-machine-learning-5f2636657d92), para servir de apoio ao pessoal que está iniciando na área. Frequentemente compartilho algum estudo legal por lá!

### Bibliografia:

Para uma leitura mais aprofundada, eu recomendo a bibliografia abaixo. O livro do Christopher M. Bishop, que cobre amplamente vários tópicos de Machine Learning, é um dos meu preferidos.

[Álgebra Linear com Aplicações — ](http://amzn.to/2zqRRLp)Anton, Howard

[Deep Learning — ](http://amzn.to/2BmFJfX)Goodfellow, Ian

[An Introduction to Statistical Learning: With Applications in R: 103](http://amzn.to/2zqSxjV) — James, Gareth

[The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition](http://amzn.to/2znbDaL) — Hastie, Trevor *et al*.

[Pattern Recognition and Machine Learning](http://amzn.to/2A7xeHq) — Christopher M. Bishop

[Hands–On Machine Learning with Scikit–Learn and TensorFlow](http://amzn.to/2CJh5pS) — Aurelien Geron

[An executive’s guide to AI](https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/an-executives-guide-to-ai)
