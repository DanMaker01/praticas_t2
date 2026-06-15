# À Fazer:

#### Modelo:

- Mudar a função objetivo, retirar a maximização do espaço usado, pois se trata de combinação, queremos um fitness que incentive coisas boas ou puna coisas ruins
  - Estabilidade resultante ao retirar um pedido
  - Ordem de posicionamento (não precisa punir tããão forte), são aceitáveis pequenos desvios
  - (...)

#### Força Bruta:

- Fazer código que gera todos e estima o tempo.
- Responder:
  - Até quando vale a pena rodar exaustivamente? Como usar isso para gerar soluções iniciais boas.

#### BRKGA:

- Controlar a punição,
  - Fazer estudos de container pequeno para fazer medição da punição
- ver o intervalo e se é linear. o fitness está muito baixo.
- Evitar fazer o fitness da mesma sequencia duas vezes usando memória temporária

#### Testes com:

- Caixas iguais
- Caixas diferentes
- 

#### Visualizador

- O visualizador deve conseguir receber soluções de um jeito rápido
- Visualização bugada
