---
layout: post
title:  "Transferência Neural de Estilos — Uma visão geral"
resume: "Recentemente, as redes neurais artificiais se mostraram capazes de transferir o estilo de uma pintura famosa para uma imagem de referência, gerando resultados visuais de encher os olhos, algo que chamaram de Transferência Neural de Estilos, capaz transformar qualquer pessoa sem talento para pintura num verdadeiro artista"
date:   2018-12-31 11:50:00
categories: [computer-vision, deep-learning]
tags: [computer-vision, deep-learning]
permalink: /deep-learning/neural-style-transfer-uma-introducao
thumbnail: "/blog/assets/img/neural-style-transfer-cover.jpg"
---

{:.image}
{:.zoom}
![Neural Style transfer](/blog/assets/img/neural-style-transfer-cover.jpg)

Recentemente, as redes neurais artificiais se mostraram capazes de transferir o estilo de uma pintura famosa para uma imagem de referência, gerando resultados visuais de encher os olhos. Logo depois que o primeiro paper abordando a transferência neural de estilos surgiu, vários outros trabalhos bem legais apareceram, transformando isto em um tópico bastante quente na comunidade de visão computacional. 

> Esta técnica pode transformar qualquer pessoa sem talento para pintura num verdadeiro artista.

É engraçado pensar que esta técnica pode transformar qualquer pessoa sem talento para pintura num verdadeiro artista, mas é verdade. A grande pergunta é: como isso é possível? 

Ao final deste post, espera-se que você tenha lido sobre os seguintes tópicos:

1. **O que significa Transferência de Estilo, ou Style Transfer?**
2. **Transferência de estilos com Redes Neurais - quem deu o primeiro passo?**
3. **Como a tranferência de estilos acontece, de um modo geral?**
4. **Alguns trabalhos legais que surgiram**
5. **Existem aplicações comerciais?**
6. **Quais são os desafios neste novo campo?**
7. **Cadê o código?**


# O que significa Transferência de Estilo, ou Style Transfer?

Já é antigo o desejo de sintetizar obras de arte no estilo de pintura a partir de imagens naturais. Em meados dos anos 1990 isto já atraia a atenção de pesquisadores. Estamos falando de um tópico de pesquisa antigo que já mostrou resultados sem o auxílio de redes neurais.

Na comunidade de visão computacional, a transferência de estilo é normalmente estudada como um problema generalizado de síntese de textura, na qual se extrai e tranfere uma textura a partir de uma imagem fonte para uma imagem alvo.

Renderizações estilizadas de imagens em 2D eram comumente realizadas com base em técnicas bem consolidadas, tais como (mas não limitadas a estas):

* **Stroke-Based Rendering** - Geralmente se inicia a partir de uma imagem fonte, com composição de traçados de maneira incremental, até que esses traçados combinem com a foto representada e produzam uma imagem que se parece com esta foto, mas com todo um estilo artístico. Uma limitação desta abordagem é que ela não é nada flexível. Cada algoritmo **SBR** só é capaz de lidar com um estilo em particular, não tendo capacidade para simular estilos arbitrários.

* **Region-Based Rendering** - Incorpora o conteúdo de determinadas regiões da imagem, de maneira segmentada. Isto permite adaptar a renderização com base nos conteúdos destas regiões. Deste modo, diferentes padrões de traçados podem ser produzidos em diferentes regiões de uma imagem, permitindo o controle local dos níveis de detalhes. Este método, contudo, também possui algumas fraquezas patentes já que a renderização baseada em regiões também não é capaz de simular estilos mais arbitrários.

* **Example-Based Rendering** - Tenta fazer um mapeamento supervisionado entre duas imagens: uma imagem fonte não estilizada e uma imagem alvo correspondente e estilizada. O algoritmo, então, aprende os padrões em torno das analogias entre estes pares de imagens no conjunto de aprendizagem e cria uma imagem estilizada análoga, dada uma imagem de testes como entrada. Esta técnica é mais eficiente em relação às duas anteriores, quando o interesse é lidar com estilos mais variados (muito embora apenas explore as features de mais baixo nível). Contudo, datasets contendo este tipo de pares mencionados são bem escassos na prática.

# Transferência de estilos com Redes Neurais - quem deu o primeiro passo?

Apesar de estes algoritmos de Renderização Artistica serem capazes de reproduzir fielmente certos tipos de estilos, eles ainda possuem limitações com respeito à sua flexibilidade, diversidade de estilos que são capazes de representar e eficiência na extração de certas estruturas das imagens.

[Gatys et al](https://arxiv.org/abs/1508.06576) mostrou, de maneira espetacular, que as redes neurais convolucionais (CNN) são capazes de codificar informações de estilo de qualquer imagem dada. Além disso, o autor mostrou que tanto o conteúdo de uma imagem de referência quanto o estilo a ser transferido de uma outra imagem podem ser separados e tratados individualmente. Como uma consequência disto, é possível transferir as características do estilo de uma dada imagem (como alguma pintura bem famosa de [Van Gogh](https://pt.wikipedia.org/wiki/Vincent_van_Gogh)) para alguma outra imagem de destino, ao mesmo tempo em que se preserva o conteúdo desta outra. Este trabalho é considerado um marco, porque deu início a um novo e empolgante campo de estudo, que se tornou um tópico de pesquisa quente, chamado: **Neural Style Transfer (NST)**. 

{:.image}
![](/blog/assets/img/neural-style-transfer-uma-introducao-gatys-et-al.png)
As imagens combinam o conteúdo de uma fotografia com o estilo de algumas pinturas bem conhecidas. Estas imagens foram criadas ao encontrar uma imagem que simultaneamente combina a representação do conteúdo da fotografia com a representação do estilo da pintura. Para mais detalhes sobre a metodologia, veja [Gatys et al](https://arxiv.org/abs/1508.06576).

Trata-se do processo de usar redes neurais convolucionais para renderizar o conteúdo de uma imagem em diferentes estilos. Este estudo atraiu muita atenção tanto da academia quanto da indústria, o que permitiu a criação de alguns produtos bem legais, como veremos mais adiante neste post.

Neste trabalho brilhante, os autores usaram 16 camadas convolucionais e 5 camadas pooling de uma rede neural [VGG-19](https://arxiv.org/pdf/1409.1556), cujos pesos já haviam sido treinados. A VGG-19 é uma rede neural convolucional treinada em mais de um milhão de imagens do [ImageNet](http://www.image-net.org). A rede possui 19 camadas profundas e é capaz de classificar imagens de objetos em **1000** categorias diferentes. Como resultado disto, a VGG-19 aprendeu riquíssimas respresentações de features de uma grande variedade de imagens.

Ao reconstruir representações de imagens a partir das camadas intermediárias de uma rede neural VGG-19 para obter uma imagem estilizada, [Gatys et al](https://arxiv.org/abs/1508.06576) observou que uma rede neural convolucional profunda é capaz de extrair o conteúdo de uma fotografia arbitrária e alguma informação de aparência de uma pintura famosa. A nova imagem estilizada é obtida por meio de um processo de otimização, no qual uma função de perda penaliza as diferenças de alto nível entre as representações obtidas a partir do conteúdo da fotografia e da imagem estilizada.

# Como a tranferência de estilos acontece, de um modo geral?

Para tranferir o estilo de uma imagem para outra, primeiro é necessário **1)** modelar e extrair o estilo de uma imagem fonte. Após obter a representação do estilo, o próximo passo é **2)** reconstruir a imagem com a informação do estilo desejado ao mesmo tempo em que se preserva seu conteúdo. 

Para conseguir alcançar o objetivo **1)**, tem-se utilizado técnicas de modelagem de textura com uso de CNNs, correlacionando features em diferentes posições da imagem, o que traz eficiência na modelagem de uma ampla variedade de texturas (naturais ou não). Já o objetivo **2)** é conseguido por meio de algumas técnicas de reconstrução de imagens.

Uma parte essencial do processo de muitas tarefas de visão computacional é extrair uma representação abstrata a partir de uma imagem de entrada. A reconstrução desta imagem é o processo reverso, o que significa reconstruir esta imagem a partir da representação previamente extraída. Novamente, uma CNN pode entrar em cena (embora não estejamos limitados apenas a isto).

Assim, a ideia básica é primeiro modelar e extrair informações de estilo e conteúdo do par de imagens, recombinar estas informações na forma de uma representação alvo e, então, obter uma imagem resultante estilizada por meio de um processo iterativo de otimização. Este é o fundamento dos algoritmos de transferência de estilos baseados na otimização de uma imagem resultante. Por este motivo, é computacionalmente custoso, mas fornece resultados espetaculares.

# Alguns trabalhos legais que surgiram

Desde que a Tranferência Neural de Estilos foi apresentada pela primeira vez, numerosos trabalhos tem sido publicados com o objetivo de melhorar esta abordagem inicial. São trabalhos que proporam melhorias em termos de velocidade de execução, resultados qualitativos e generalização para vários outros estilos.

Surgiram alguns trabalhos bem legais que tentam fazer transferência de estilos para obter fotos realistas, como em [Luan et al.(2017)](https://arxiv.org/pdf/1703.07511.pdf), que transmitem o estilo de uma imagem fonte para uma imagem alvo, deixando esta última tão realista quanto possível. A escolha apropriada da foto de referência pode fazer com que a foto para a qual se deseja transferir o estilo pareça ter sido obtida com uma iluminação diferente, numa hora do dia ou condição climática diferentes, ou que ela pareça ter sido retocada com alguma outra intenção.

{:.image}
{:.zoom}
![Deep Photo Style Transfer - Luan et al.(2017)](/blog/assets/img/neural-style-transfer-uma-introducao-luan-et-al_2.png)
Este é o resultado da proposta de [Luan et al.(2017)](https://arxiv.org/pdf/1703.07511.pdf), cujo código pode ser [encontrado no Github](https://github.com/luanfujun/deep-photo-styletransfer), e permite fazer transferência de estilos para obter fotos realistas. Baseia-se no trabalho [Gatys et al.](https://arxiv.org/abs/1508.06576). **Clique para ampliar**

Um outro trabalho bem bacana foi o de [Pęśko, M & Trzciński, T (2018)](https://arxiv.org/pdf/1809.01726). Neste paper, os autores avaliam e comparam os resultados obtidos por meio de vários métodos no contexto de transferência de estilo de quadrinhos, focando principalmente na otimização do tempo de execução por imagem. Mais precisamente, eles comparam vários métodos de transferência de estilos e avaliam sua eficiência em termos de quão bem eles propagam várias características do estilo de quadrinhos entre as imagens.

{:.image}
{:.zoom}
![Neural Comic Style transfer](/blog/assets/img/neural-comic-style-transfer.png)
Resultados da aplicação de diferentes métodos de transferência de estilos no contexto de quadrinhos. A primeira coluna é a imagem de estilo; a segunda coluna contém as imagens de referência - [Pęśko, M & Trzciński, T (2018)](https://arxiv.org/pdf/1809.01726). **Clique para ampliar**

[Pegios et al.(2018)](https://arxiv.org/pdf/1811.12704) argumenta que fazer uso dos sub-estilos presentes em uma imagem de estilo pode melhorar o resultado final do processo de TNE. A principal contribuição do paper é um método para modelar separadamente cada sub-estilo que existe numa imagem de estilo e, então, combinar cada um desses sub-estilos na região mais apropriada da imagem de referência, estilizando cada região correspondente da imagem de referência com um sub-estilo específico (brilhante!). Para isto, o método detecta e decompõe todos os sub-estilos que existem na imagem de estilo, como mostrado na figura abaixo, além de segmentar o conteúdo da imagem de referência em várias regiões semânticas usando um modelo de mistura de gaussianas.

{:.image}
{:.zoom}
![Style Decomposition for Improved Neural Style Transfer - Pegios et al.(2018)](/blog/assets/img/style-decomposition-for-improved-neural-style-transfer.png)
Decomposição de estilo e conteúdo. À esquerda, temos a imagem de estilo original e imagem de referência. As outras imagens à direita correspondem às mascaras de sub-estilo e sub-conteúdos detectados pelo método proposto. Esta imagem foi retirada de [Pegios et al.(2018)](https://arxiv.org/pdf/1811.12704)

Há outros trabalhos bem legais envolvendo TNE no contexto de vídeos, como em [Ruder et al.(2016)](https://arxiv.org/pdf/1604.08610). Neste paper, os autores abordam um método que transfere o estilo de uma imagem (como o de uma pintura, por exemplo) para uma sequência de frames, propondo uma nova função de perda que permite gerar vídeos estilizados com boa estabilidade, mesmo quando as imagens são oclusas. No vídeo abaixo é possível conferir o resultado dos experimentos realizados pelos autores do paper:

<div class="video-container">
	<iframe width="560" height="315" src="https://www.youtube.com/embed/vQk_Sfl7kSc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

# Existem aplicações comerciais?

Em virtude do visual exuberante das imagens estilizadas, a pesquisa em Transferência Neural de Estilos levou à aplicações interessantes na indústria e já entrega benefícios comercialmente. Uma razão pela qual a TNE cativa os olhos tanto na academia quanto na indústria é sua popularidade em redes sociais como Facebook, Twitter e Instagram. 

[Prisma](https://prisma-ai.com/) foi uma das primeiras aplicações de software a oferecer um algoritmo de TNE como serviço, tendo alcançado bastante sucesso pelo mundo e, inclusive, recebendo algumas premiações nas *app stores*. Outros vieram depois, com objetivos similares, como o [Ostagram](https://www.ostagram.me/). Com o auxílio destes aplicativos, as pessoas podem criar suas próprias artes e compartilhá-las nas redes sociais. Se você sempre quis ser um artista como o Van Gogh, está aí a sua oportunidade.

{:.image}
{:.zoom}
![Made with Prisma](/blog/assets/img/neural-style-transfer-prisma.png)
Algumas imagens obtidas com o aplicativo [Prisma](https://prisma-ai.com/), que usa Transferência Neural de Estilos como um serviço e já recebeu alguns prêmios.

{:.image}
{:.zoom}
![Transferência Neural de Estilo com Ostagram](/blog/assets/img/neural-style-transfer-ostagram.jpg)
Imagens estilizadas com a ferramenta [Ostagram](https://www.ostagram.me)

Há algumas outras aplicações possíveis para estas técnicas, como na indústria do entretenimento: filmes, animações, jogos, etc.

# Quais são os desafios neste novo campo?

As abordagens existentes no campo da transferência de estilos frequentemente sofrem de um *trade-off* entre generalização, qualidade e eficiência. Os métodos baseados em otimização da representação das imagens são capazes de manipular estilos arbitrários com uma qualidade visual superior àquela obtida por meio de outros métodos, mas são computacionalmente intensivos.

Outro problema gira em torno do fato de não haver um conjunto de imagens de benchmark padrão para avaliar os algoritmos. Normalmente os autores utilizam suas próprias imagens para avaliação, comparando imagens lado a lado, ou conduzem estudos com usuários com o objetivo de analisarem a qualidade dos seus trabalhos com base no voto que cada um dá às imagens estilizadas.

Outra dificuldade existe com relação à interpretabilidade destes algoritmos neurais. Como muitas tarefas de visão computacional baseadas em CNN, a TNE é como uma caixa-preta também. Contudo, como estamos falando de tópico de pesquisa que está bastante aquecido, espera-se que novos trabalhos sejam publicados com propostas que permitam cortornar estes problemas citados.

# Cadê o código? Sem código não é divertido!

Eu disponibilizei o código fonte em [tensorflow](https://www.tensorflow.org/), que utilizei durante meus experimentos **(figura abaixo)**, e você já pode rodar ele direto no [Google Colab](https://colab.research.google.com/drive/1ZDcuRwbIpu3lA_83VWwpOBAjhxXJUSM-) (não esquece de habilitar a GPU). Use-o como um ponto de partida para que você possa avançar em seus estudos com Transferência Neural de estilos, inserindo melhorias com o objetivo de obter resultados melhores. Você vai notar que eu não utilizei uma VGG, mas sim uma [GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) (vencedora do ILSVRC 2014, ficando nos top-5 com taxa de erro de 6.67%, superando a VGG), mas você pode tentar usar uma VGG no lugar. Na imagem abaixo você confere o resultado que consegui obter em meus experimentos.

{:.image}
{:.zoom}
![Transferência Neural de Estilos](/blog/assets/img/neural-style-transfer-experimento-tensorflow.jpg)
Resultado de um experimento que conduzi usando Tensorflow. **(1)** é uma foto que tirei em Olinda-PE há algum tempo e usei como imagem de referência no experimento; **(2)** é uma [pintura que peguei na internet](https://www.carredartistes.com/us/en/art-online-gallery-contemporary-artist-frederic-thiery/14350-unique-contemporary-artwork-frederic-thiery-new-york-city.html); **(3)** é o resultado da transferência do estilo presente na imagem (2) para a imagem (1).

Este post será atualizado em breve, com links para novos códigos, conforme eu for testando outras implementações.

## Referências, para um estudo mais aprofundado

[A Neural Algorithm of Artistic Style - Gatys et al.(2015)](https://arxiv.org/abs/1508.06576)

[Deep Photo Style Transfer - Luan et al.(2017)](https://arxiv.org/pdf/1703.07511.pdf)

[Neural Comic Style Transfer: Case Study - Pęśko, M & Trzciński, T (2018)](https://arxiv.org/pdf/1809.01726)

[Neural Style Transfer: A Review - Jing et al.(2017)](https://arxiv.org/pdf/1705.04058)

[Style Decomposition for Improved Neural Style Transfer - Pegios et al.(2018)](https://arxiv.org/pdf/1811.12704)

[Artistic style transfer for videos - Ruder et al.(2016)](https://arxiv.org/pdf/1604.08610)

[Very Deep Convolutional Networks for Large-Scale Image Recognition - Simonyan, & Zisserman(2014)](https://arxiv.org/pdf/1409.1556) - ***VGGNet***

[Going Deeper with Convolutions - Szegedy1 et al. (2015)](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) - ***GoogLeNet***
