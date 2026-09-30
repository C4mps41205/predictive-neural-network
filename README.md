# Rede neural do zero: reconhecendo dígitos escritos à mão

## 1. O que queremos fazer?

Queremos ensinar um programa a **olhar para a imagem de um número escrito à mão (0 a 9) e dizer qual é o número**.

Para nós, isso é fácil. Para o computador, uma imagem é só uma tabela de números. Então, o que fazemos é:

1. transformar a imagem em números;
2. passar esses números por uma "máquina de fazer contas" (a rede neural);
3. ler a resposta que sai no final;
4. comparar com a resposta certa e **ajustar a máquina** para errar menos da próxima vez.

Repetindo isso muitas vezes, a máquina "aprende".

---

## 2. Como o computador enxerga a imagem

Cada imagem tem **28 × 28 = 784 pixels**. Cada pixel é um número de 0 (preto) a 255 (branco), que indica o tom de cinza.

Colocamos todos os pixels de uma imagem em fila, formando uma lista de 784 números. Se temos $m$ imagens, empilhamos as listas lado a lado (uma imagem por coluna), formando a matriz $X$:

$$X \in \mathbb{R}^{784 \times m}$$

Também temos $Y$, a lista das respostas certas (o dígito verdadeiro de cada imagem).

> **Dica:** em vez de usar valores de 0 a 255, costuma-se dividir tudo por 255 para que fiquem entre 0 e 1. Isso deixa as contas mais estáveis.

---

## 3. A arquitetura: as três camadas

Nossa rede tem uma estrutura simples, com **duas camadas de contas** e três "andares":

```
 Camada 0 (entrada)      Camada 1 (oculta)        Camada 2 (saída)
    784 números    →       10 neurônios      →       10 neurônios
   (os pixels)          (função ReLU)              (função softmax)
                                                   (um para cada dígito)
```

- **Entrada** $A^{[0]} = X$: os 784 pixels da imagem.
- **Camada oculta** $A^{[1]}$: 10 "neurônios" que procuram padrões simples (traços, curvas, cantos).
- **Saída** $A^{[2]}$: 10 neurônios, um para cada dígito de 0 a 9. O que tiver o maior valor é a resposta da rede.

Um **neurônio** aqui é só uma pequena conta: ele pega os números que chegam, multiplica cada um por um "peso" (a importância daquela entrada), soma tudo e adiciona um valor extra chamado *viés*.

---

## 4. Os "botões" da máquina: pesos e vieses

A rede tem botões que podem ser ajustados. Esses botões são os **parâmetros**:

| Parâmetro | Significado | Formato |
|-----------|-------------|---------|
| $W^{[1]}$ | pesos que ligam os 784 pixels aos 10 neurônios ocultos | $10 \times 784$ |
| $b^{[1]}$ | viés dos 10 neurônios ocultos | $10 \times 1$ |
| $W^{[2]}$ | pesos que ligam os 10 neurônios ocultos aos 10 de saída | $10 \times 10$ |
| $b^{[2]}$ | viés dos 10 neurônios de saída | $10 \times 1$ |

No começo, todos são **sorteados aleatoriamente** (valores entre −0,5 e 0,5). A rede começa "chutando" e vai melhorando aos poucos.

---

## 5. Forward propagation: fazendo uma previsão

"Forward" quer dizer "para a frente": os números entram pela esquerda e vão passando pelas camadas até sair a resposta.

### Passo 1: primeira conta (pixels → camada oculta)

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

Cada neurônio oculto faz uma **soma ponderada** dos pixels. É como uma votação em que cada pixel tem um peso diferente na decisão.

### Passo 2: função de ativação ReLU

$$A^{[1]} = g_{ReLU}\left(Z^{[1]}\right), \qquad g_{ReLU}(z) = \max(0, z)$$

A ReLU é simples: **se o número for negativo, vira zero; se for positivo, continua igual.** Ela funciona como um interruptor: o neurônio só "liga" se a soma for positiva.

Por que precisamos disso? Sem essa etapa, a rede inteira seria apenas uma grande conta de multiplicar e somar, e não conseguiria aprender formas complicadas como um "8" ou um "5".

### Passo 3: segunda conta (camada oculta → saída)

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

É o mesmo tipo de conta, agora usando a saída da camada oculta como entrada.

### Passo 4: função softmax

$$A^{[2]} = g_{softmax}\left(Z^{[2]}\right), \qquad \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

A softmax transforma os 10 números da saída em **probabilidades**: todos ficam entre 0 e 1 e somam 1. Por exemplo:

| Dígito | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|--------|---|---|---|---|---|---|---|---|---|---|
| Probabilidade | 0,01 | 0,02 | 0,01 | 0,03 | 0,02 | 0,01 | 0,02 | **0,85** | 0,02 | 0,01 |

Aqui a rede diz: "tenho 85% de certeza de que é um **7**". A previsão é o dígito de maior probabilidade.

---

## 6. Medindo o erro

Como saber o quanto a rede errou? Primeiro, transformamos a resposta certa em uma lista de zeros com um único 1 (técnica chamada *one-hot*). Se a imagem é um **3**:

$$Y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$$

Depois comparamos com a previsão $A^{[2]}$. Quanto mais diferente, maior o erro. Uma medida usada para isso é a **entropia cruzada**:

$$L = -\sum_{i=0}^{9} y_i \log(a_i)$$

Ela é grande quando a rede dá pouca probabilidade ao dígito correto e é próxima de zero quando a rede acerta com confiança. **O objetivo do treino é fazer esse erro diminuir.**

---

## 7. Backward propagation: descobrindo quem errou

Agora vem a parte mais importante. Precisamos descobrir **o quanto cada botão (peso e viés) contribuiu para o erro**, para saber em que direção ajustá-lo.

Fazemos isso caminhando de trás para frente (por isso "backward"): começamos pela saída e voltamos até a entrada. Essa matemática usa a *regra da cadeia* do cálculo, e o resultado é um conjunto de fórmulas simples.

Usamos a letra $d$ na frente para indicar "o quanto o erro muda se esse valor mudar" (o **gradiente**).

### Erro da camada de saída

$$dZ^{[2]} = A^{[2]} - Y$$

É a previsão menos a resposta certa. Se a rede previu 0,85 para o 7 mas o certo era 3, o erro fica grande nesses dois dígitos.

### Ajuste da segunda camada

$$dW^{[2]} = \frac{1}{m}\, dZ^{[2]}\, {A^{[1]}}^{T}$$

$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

O $\frac{1}{m}$ tira a **média** sobre todas as $m$ imagens, para que o ajuste reflita o conjunto todo e não uma imagem só.

### Levando o erro para trás

$$dZ^{[1]} = {W^{[2]}}^{T}\, dZ^{[2]} \;\odot\; g^{[1]\prime}\left(Z^{[1]}\right)$$

Aqui o erro "volta" pela camada oculta. O símbolo $\odot$ significa multiplicar elemento por elemento. E $g'$ é a derivada da ReLU:

$$g'_{ReLU}(z) = \begin{cases} 1 & \text{se } z > 0 \\ 0 & \text{se } z \le 0 \end{cases}$$

Ou seja: neurônios que estavam "desligados" (zero) não recebem culpa nem ajuste, porque não influenciaram a resposta.

### Ajuste da primeira camada

$$dW^{[1]} = \frac{1}{m}\, dZ^{[1]}\, {A^{[0]}}^{T}$$

$$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

Lembre que $A^{[0]} = X$, a própria imagem de entrada.

---

## 8. Atualizando os parâmetros: aprendendo

Com os gradientes em mãos, damos um **pequeno passo** em cada botão, na direção que diminui o erro:

$$W^{[2]} := W^{[2]} - \alpha\, dW^{[2]}$$

$$b^{[2]} := b^{[2]} - \alpha\, db^{[2]}$$

$$W^{[1]} := W^{[1]} - \alpha\, dW^{[1]}$$

$$b^{[1]} := b^{[1]} - \alpha\, db^{[1]}$$

O símbolo $:=$ significa "passa a valer". O $\alpha$ (alfa) é a **taxa de aprendizado**, o tamanho do passo:

- **muito pequeno:** a rede aprende, mas demora muito;
- **muito grande:** a rede "passa do ponto" e fica instável.

Uma boa imagem é descer uma montanha com neblina: você sente a inclinação do chão (o gradiente) e dá um passo para baixo, repetidamente, até chegar ao vale (o menor erro). Esse método chama-se **gradiente descendente**.

---

## 9. O ciclo completo de treino

Repetimos, por exemplo, 500 vezes:

1. **Forward:** calcular $Z^{[1]}, A^{[1]}, Z^{[2]}, A^{[2]}$ (fazer a previsão);
2. **Backward:** calcular $dZ^{[2]}, dW^{[2]}, db^{[2]}, dZ^{[1]}, dW^{[1]}, db^{[1]}$ (descobrir os erros);
3. **Atualizar:** ajustar $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$ (aprender).

A cada rodada, a acurácia (porcentagem de acertos) deve subir.

---

## 10. Tabela de formatos das matrizes

Conferir os formatos ajuda a entender e a depurar o código. Aqui, $m$ é o número de imagens.

### Forward propagation

| Variável | Formato | Observação |
|----------|---------|------------|
| $A^{[0]} = X$ | $784 \times m$ | pixels de entrada |
| $W^{[1]}$ | $10 \times 784$ | precisa casar com $A^{[0]}$ na multiplicação |
| $b^{[1]}$ | $10 \times 1$ | somado a todas as colunas |
| $Z^{[1]}, A^{[1]}$ | $10 \times m$ | camada oculta |
| $W^{[2]}$ | $10 \times 10$ | precisa casar com $A^{[1]}$ na multiplicação |
| $b^{[2]}$ | $10 \times 1$ | somado a todas as colunas |
| $Z^{[2]}, A^{[2]}$ | $10 \times m$ | saída (probabilidades) |

### Backward propagation

| Variável | Formato | Mesmo formato de |
|----------|---------|------------------|
| $dZ^{[2]}$ | $10 \times m$ | $A^{[2]}$ |
| $dW^{[2]}$ | $10 \times 10$ | $W^{[2]}$ |
| $db^{[2]}$ | $10 \times 1$ | $b^{[2]}$ |
| $dZ^{[1]}$ | $10 \times m$ | $A^{[1]}$ |
| $dW^{[1]}$ | $10 \times 784$ | $W^{[1]}$ |
| $db^{[1]}$ | $10 \times 1$ | $b^{[1]}$ |

**Regra útil:** o gradiente de um parâmetro sempre tem o **mesmo formato** que o próprio parâmetro, porque cada peso precisa do seu ajuste.

---

## 11. Resumo em uma frase por etapa

| Etapa | Em linguagem simples |
|-------|----------------------|
| Entrada $X$ | a imagem transformada em números |
| $Z = WA + b$ | soma ponderada: cada entrada tem uma importância |
| ReLU | liga o neurônio só se o resultado for positivo |
| Softmax | transforma os resultados em probabilidades |
| $dZ^{[2]} = A^{[2]} - Y$ | previsão menos resposta certa = o erro |
| Backprop | espalha o erro de volta para descobrir quem errou |
| $W := W - \alpha\, dW$ | corrige cada peso um pouquinho |
| Repetir | quanto mais repetições, menor o erro |
