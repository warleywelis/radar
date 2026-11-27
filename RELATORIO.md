# Radar Binance: relatório completo (situação em 07/11/2026)

Este documento resume tudo o que o projeto é, tudo o que foi feito, o que se descobriu e o que falta. Foi escrito para você (ou o Claude, numa conversa nova) retomar sem precisar repetir nada.

---

## 1. Em uma página

- **O que é:** um robô de SINAIS e estudo para a Binance (mercado à vista), só de simulação, com capital de R$ 50. Ele roda sozinho no GitHub, de hora em hora, e um painel no celular mostra o que ele fez. Não opera dinheiro de verdade e não tem acesso à sua conta.
- **Onde está:** repositório `warleywelis/radar` no GitHub; painel em https://warleywelis.github.io/radar/ (menu), https://warleywelis.github.io/radar/nuvem.html (painel do robô) e https://warleywelis.github.io/radar/cenarios.html (cenários de 7 dias).
- **Onde estamos:** os testes históricos mostram uma vantagem pequena do modelo (cerca de R$ 0,8 (1,6%) por operação), mas **na prática (simulação ao vivo) o resultado até agora é negativo**: operação principal com 7 fechadas, 3 lucros e total de R$ -10,20, e as regras de comparação também perdem. O mercado estava em queda (Bitcoin caiu 7,3% em outubro e entrou em novembro perto de US$ 78 mil).
- **Veredito honesto:** ainda não há evidência de que dá lucro de verdade. A vantagem do teste histórico não apareceu na prática até agora. O correto é continuar coletando sem mexer nas regras e **não usar dinheiro real**. Mesmo que um dia funcione, com R$ 50 são centavos por operação.
- **O que fazer agora:** enviar os arquivos novos ao GitHub, rodar a pesquisa e o robô, e olhar o painel uma vez por semana (seção 10).

---

## 2. Objetivo, regras combinadas e restrições

- Objetivo inicial: com R$ 50, buscar lucro líquido modesto e rápido, já descontadas as taxas, estudando o preço e as opções na Binance. A IA não adivinha o preço; o projeto virou uma máquina de **testar ideias com honestidade**.
- Custos considerados: taxa de 0,075% por ordem (pagando em BNB; 0,1% sem BNB), slippage de 0,1% por ordem, conversão entre reais e USDT de 0,1% por conversão (só no começo e no fim, porque o dinheiro fica em USDT entre as operações). R$ 50 equivalem a cerca de 9,84 USDT (com o dólar digital a R$ 5,08 em outubro). O mínimo de ordem da Binance é algo como 5 dólares (uns R$ 25 a R$ 30).
- **Critério para pensar em dinheiro real:** pelo menos 30 operações-sombra fechadas, média positiva depois de todos os custos e vantagem segura contra entradas ao acaso (z de 3 ou mais quando várias regras foram testadas). Mesmo então: começar com o mínimo, só dinheiro que aceita perder, sem alavancagem, e nunca colar chaves de API em conversa. Ganhos em criptoativos têm regras de declaração na Receita. O Claude não é consultor financeiro.
- Formato dos valores nas respostas e no painel: `R$ 2,50 (5,0%)`. Português do Brasil, prosa simples, sem emojis.
- Seu ambiente: só o celular (Android, Chrome, Termux disponível). O PC do trabalho não permite instalar nada. Nunca usar o PC do trabalho para isso.
- Você prefere respostas curtas, sem repetir instruções (isso foi salvo nas preferências).

---

## 3. Como o sistema funciona

### 3.1 Peças e fluxo de dados
- **GitHub Actions** roda dois fluxos: `robo` (de hora em hora) e `pesquisa` (manual, e semanal automático). Os resultados ficam em arquivos `data/*.json` dentro do repositório. A página lê esses arquivos pelo `raw.githubusercontent.com`. Por isso a página só mostra, não calcula nada pesado.
- **Fontes de dados:** `data-api.binance.vision` (API pública só de mercado; o `api.binance.com` é bloqueado nos servidores do GitHub, que ficam nos EUA) e `data.binance.vision` (arquivo público de histórico, que inclui moedas que já saíram da Binance).
- **Cada mensagem do Claude reprocessa a conversa inteira**, por isso este relatório existe: para trabalhar em conversas novas dentro de um Projeto.

### 3.2 O sinal (momentum)
Rompimento da máxima dos 20 dias anteriores com fechamento acima dela e volume pelo menos 2 vezes a média de 20 dias, em candles diários, nas 60 moedas de maior volume médio de 30 dias (ranking igual ao do treino). Excluídos: stablecoins, ouro, tokens alavancados (UP/DOWN/BULL/BEAR) e qualquer moeda com oscilação diária (ATR) abaixo de 1,2%.

### 3.3 A operação simulada
- Entrada no preço de compra do momento (+0,1% de slippage), na primeira hora depois do candle diário fechar (21h de Brasília).
- Meta de 10%. Stop de 2 oscilações diárias (entre 4% e 15%). Prazo de 7 dias. Uma operação por vez na carteira principal.
- A operação é conferida de hora em hora pelas velas de 1 hora. O lucro mostrado na página é o preço de agora; **meta e stop só são registrados na próxima execução do robô** (foi o caso do IOST: a página chegou a mostrar R$ 31, e o registro foi R$ 4,93 (9,9%), que é o que uma ordem de venda programada na meta teria feito).
- **Sinal esticado:** se o preço já subiu mais de 1,5 oscilação desde o sinal, o robô não entra (não corre atrás do preço).
- **Pausa:** depois de 3 stops seguidos, as entradas pausam por 3 dias.

### 3.4 O modelo dos 3 caminhos
- Prevê, no fechamento do dia, onde o preço estará em 7 dias: **alta** (subir pelo menos 1 oscilação), **estabilidade** ou **baixa** (cair pelo menos 1 oscilação). Regressão logística multinomial feita à mão (em `ml.js`), sem bibliotecas.
- Atributos (12 básicos): variações de 1, 3, 7 e 20 dias, distância da máxima de 20 dias, volume, tendência, inclinação da média de 50 dias, nível de oscilação, corpo do candle, retorno de 7 dias e tendência do Bitcoin (tudo em unidades da oscilação da moeda). Variante com 15: mais RSI, ADX e volume de 5 dias.
- **Validação:** teste cronológico mês a mês (treina só com o passado, com intervalo de segurança de 8 dias, prevê o mês seguinte). O limite de sinal (chance de alta de 40% ou mais) é escolhido na primeira parte do período fora da amostra e conferido na parte final, que nunca foi usada para escolher nada. O modelo só é "validado" se: o erro de previsão for menor que o do chute básico, houver pelo menos 40 operações, a média for positiva e a diferença contra entradas ao acaso tiver z de 2 ou mais.
- **Aprendizado contínuo:** cada previsão é guardada; 8 dias depois, o robô compara com o resultado real e ajusta o modelo um pouco, mantendo um placar ao vivo. Se, depois de 100 previsões, o modelo errar mais que o chute básico, ele é rebaixado e o robô volta à regra de momentum.
- **Variantes testadas a cada pesquisa** (poucas de propósito): top 60 básico (em uso), top 60 com RSI/ADX/volume 5 dias, top 40, top 80, top 100, persistência de 7 dias no ranking, e a nova "prevendo o resultado da operação" (meta antes do stop). Só se troca de variante com z de 3 ou mais contra a base no mesmo período.

### 3.5 As barras de força (1 a 5)
Nota do sinal que combina volume, quão acima da máxima o preço fechou, tendência, estado do Bitcoin e a distância entre o preço de agora e o fechamento do sinal. As barras são quintis dessa nota entre os sinais históricos, calibrados com os 60% mais antigos e testados nos 40% mais recentes. A página recalcula com o preço de agora. Servem para **filtrar** (evitar 1 e 2 barras), não para ajustar a meta.

### 3.6 Outras pesquisas do fluxo `pesquisa`
- **Regras de saída:** meta fixa, sem meta, stop móvel, metas menores, prazos de 3 e 14 dias, etc., sempre contra entradas ao acaso.
- **Força e carteira de R$ 50:** simula comprar, fechar no lucro e reabrir no melhor sinal com barras suficientes, uma operação por vez, e a "regra dinâmica" (meta que muda com as barras).
- **Detector de véspera:** testa prever, no fechamento de um dia, se o dia seguinte será uma vela de alta forte (1,5 oscilação e pelo menos 5%). Só vale se aumentar a taxa de acerto em pelo menos 2 vezes, com z de 3 ou mais.
- **Conferência:** refaz, pelas regras do teste histórico, os sinais que o robô já fechou, para separar se a diferença entre teste e prática vem da execução do robô ou dos sinais e do mercado.

### 3.7 Proteções que já existem
Filtro de oscilação mínima e lista de moedas atreladas; aviso vermelho no painel quando algum resultado vem de versão antiga dos arquivos; o robô ignora um modelo de versão incompatível; sinal esticado nunca entra; pausa depois de 3 stops; operações-sombra não duplicam; `concurrency` nos fluxos para não rodarem ao mesmo tempo.

### 3.8 O que o painel (`nuvem.html`) mostra
Operação aberta (com força na entrada e aviso quando stop ou meta já foram atingidos), sinais da última execução (com barras ao vivo, gráfico do TradingView em tela cheia e atalho para o TradingView com a sua conta, e o motivo de entrar ou não), modelo e variantes, força do sinal e carteira, detector de véspera, resumo da operação principal, operações-sombra por regra (contando também as abertas pelo preço de agora), conferência, histórico, regras de saída e teste de sobrevivência.

---

## 4. O que fizemos, em ordem

1. **Descoberta:** taxas da Binance, mínimo de ordem, ferramentas existentes (Freqtrade, ccxt etc.), decisão de construir uma ferramenta própria de estudo, com entrevista e passos pequenos.
2. **Ferramentas no celular (arquivos HTML):** radar de moedas com custo real, stop sugerido pela oscilação (ATR), backtests (queda, momentum de 1 hora, 7 dias, filtros, robustez) e simulador sem dinheiro. Problema: o Chrome perde o arquivo ao recarregar (endereço `content://`).
3. **Hospedagem:** Tampermonkey no Firefox não funcionou; solução definitiva com **GitHub Pages**.
4. **Robô na nuvem (GitHub Actions):** registra operações simuladas sozinho, com operações-sombra de várias regras e teste de viés de sobrevivência (moedas que saíram da Binance).
5. **Modelo, barras, saídas e carteira:** modelo dos 3 caminhos com validação cronológica; barras de força; testes de saída; carteira de R$ 50; regra dinâmica por barras; cenários de 7 dias.
6. **Correções importantes:** filtro de moedas atreladas ao dólar (distorciam treino e testes); entender que o lucro na página é o preço de agora; sinal esticado (caso BMT); viés das operações "sem meta" (as fechadas só mostravam as perdas); robô de hora em hora (o agendamento diário do GitHub atrasava horas); avisos de versão antiga.
7. **Consolidação:** `pesquisa.js` (treino, saídas, força, véspera e conferência em uma só execução, baixando o histórico uma vez), menu enxuto, PROJETO.md, preferência de respostas curtas.
8. **Últimas ideias testadas:** variantes de universo e atributos, detector de véspera, conferência do robô contra o teste histórico, rótulo "resultado da operação".

---

## 5. Resultados dos testes históricos (com o filtro de moedas atreladas)

**Modelo dos 3 caminhos (retreinado em 07/11 com 160425 exemplos de 604 moedas, 147820 previsões fora da amostra):**
- Validado. Sinais do modelo (chance de alta de 40% ou mais): 261 operações, média R$ 0,82 (1,6%) ± R$ 0,51 (1,0%), contra R$ -0,37 (-0,7%) das entradas ao acaso, z = 4,54.
- Erro de previsão 0,986 contra 1,010 do chute básico (2,3% de melhora), mas o acerto do caminho mais provável fica abaixo do básico (53,5% contra 54,1%). O valor do modelo está em ordenar as moedas.
- Calibração boa até 40% e ruim acima de 50% (previu 50% a 60%, aconteceu em 23%): não tratar probabilidades altas como literais.
- Regra de momentum pura no mesmo período: 524 operações, média R$ 0,27 (0,5%).
- Variantes (28/10): top 40, 60, 80, com RSI/ADX/volume de 5 dias e persistência de 7 dias passaram todas, mas nenhuma é melhor que a base com z de 3. Top 100 não passou. Exigir momentum junto não melhorou.

**Regras de saída (sobre os sinais de momentum, ~2150 operações):**
- Meta de 10% com stop de 2 oscilações: R$ 0,25 (0,5%), 58% de acerto.
- Sem meta (stop de 2 oscilações e prazo de 7 dias): R$ 0,72 (1,4%) ± R$ 0,54, mas só 38% de acerto.
- Meta de 5%, stop curto e meta proporcional à oscilação: perdem dinheiro. Stop móvel: R$ 0,11 (0,2%).
- Quando as operações com meta ganham, ganham R$ 4,93; quando o stop bate, perdem R$ 7,62: é preciso acertar mais de 60% das vezes só para empatar.

**Barras de força:** separam sinais bons dos ruins (3 a 5 barras: R$ 1,80 (3,6%); 1 a 2 barras: R$ -1,23 (-2,5%); z = 3,9), mas não melhoram de forma regular entre 3, 4 e 5. Na carteira de R$ 50, aceitar qualquer sinal arruinou o capital (R$ 50 viraram R$ 1 a R$ 5), e exigir 3 barras manteve no azul, com quedas de 29% a 61% no caminho.

**Regra dinâmica por barras (metas 3, 4, 5, 8 e 10%):** nenhuma variante passou e nenhuma bateu a meta fixa de 10% (R$ 0,32 a R$ 0,45 por operação contra R$ 0,60).

**Viés de sobrevivência:** pequeno. Momentum com as moedas que saíram: R$ 0,32 (0,6%); só com as de hoje: R$ 0,35 (0,7%).

**Ainda não medidos nos dados reais:** o detector de véspera, a conferência e a variante "resultado da operação" (precisam do novo `pesquisa.js` rodando).

---

## 6. Placar real da simulação (até 07/11)

**Operação principal (a que o robô "opera"):** 7 fechadas, 3 lucros.
| Moeda | Resultado |
|---|---|
| AVA | stop, R$ -7,62 (-15,2%) |
| IOST | meta, R$ 4,93 (9,9%) |
| ALT | stop, R$ -7,62 (-15,2%) |
| BMT | stop, R$ -7,62 (-15,2%) |
| INIT | meta, R$ 4,93 (9,9%) |
| BMT (2ª vez) | meta, R$ 4,93 (9,9%) |
| TRX | stop, R$ -2,12 (-4,2%) |

Total R$ -10,21; média R$ -1,46 (-2,9%) ± R$ 4,75 (9,5%).

**Operações-sombra (todos os sinais das regras):**
- Momentum com meta de 10%: 17 fechadas, 7 lucros (41%), média R$ -2,12 (-4,2%) ± R$ 3,01.
- Momentum sem meta: 16 fechadas, 1 lucro (6%), média R$ -5,50 (-11,0%) ± R$ 3,28 (quase todas já fechadas; a chance de isso ser azar é de cerca de 0,5%).
- Barras 3+ (janela de 3 dias): 5 fechadas, 2 lucros, média R$ -1,50 (-3,0%).
- Modelo de previsão: 10 fechadas, 0 lucros, todas paradas em R$ -0,37 (-0,7%) (moedas calmas, onde a meta de 10% nunca chega).
- Modelo com 3 barras, modelo e momentum, modelo fora do universo: quase sem operações ainda.

---

## 7. Por que a prática está pior que o teste (hipóteses)

1. **Mercado em queda:** Bitcoin caiu 7,3% em outubro e entrou em novembro perto de US$ 78 mil; rompimentos falham mais assim. O teste histórico mistura todas as fases.
2. **Moedas das primeiras semanas:** até o fim de outubro o robô escolhia as 60 moedas pelo volume do dia (inclui moedas explodidas no próprio dia), e não pelo volume médio de 30 dias como no teste. Entraram moedas pequenas que sobem e devolvem. O robô novo corrige isso, mas as operações antigas continuam no placar.
3. **Desalinhamento do modelo:** ele prevê "subir 1 oscilação", o que é fácil para moedas calmas (BTC, BNB, TRX), enquanto o robô ganha com meta fixa de 10%. A variante "resultado da operação" tenta alinhar as duas coisas.
4. **Execução do robô** (hora e preço de entrada): a etapa "conferência" mede isso.
5. **Possível otimismo do teste histórico:** não pode ser excluído.

---

## 8. Limites e avisos

- O simulador é mais otimista que a vida real: assume a venda exatamente no preço da meta, spread de 0,1% mesmo em moedas pequenas e stop respeitado. Em quedas rápidas o stop real costuma sair pior.
- Poucas operações dão resultados muito instáveis (com 7 operações, uma ou duas decidem tudo). Cada vez que se testa mais uma variante, aumenta a chance de uma parecer boa por sorte.
- O mínimo de ordem da Binance já consome metade de R$ 50, o que limita a uma posição por vez.
- Moedas pequenas têm "pump e dump": sobem e devolvem tudo em horas.
- Nada aqui é recomendação de investimento.

---

## 9. Próximos passos (em ordem)

1. Enviar ao repositório os arquivos da seção 10 e criar/atualizar os fluxos.
2. Rodar o fluxo `pesquisa` (40 a 60 minutos) e depois o `robô`.
3. No painel, conferir: o aviso vermelho de versão deve ficar vazio; ler a seção **Conferência**, o **Detector de véspera** e a variante **"resultado da operação"** na tabela de variantes.
4. Deixar o robô coletar por 3 a 4 semanas sem mexer nas regras (olhar o painel uma vez por semana). Não trocar regra com menos de 30 operações-sombra fechadas por coorte.
5. Comparar quando houver dados: modelo sozinho, modelo com 3 barras, modelo com momentum, moedas fora do universo.
6. Só trocar a regra de entrada com média positiva depois dos custos e z de 3 ou mais.
7. Opcional: alertas por Telegram.

---

## 10. Passo a passo no celular

### 10.1 Arquivos do repositório (versão final)
Enviar por **Add file, Upload files** (nomes iguais substituem os antigos), todos na pasta principal:
- `bot.js`, `ml.js`, `dados.js`, `pesquisa.js`
- `nuvem.html`, `index.html`, `cenarios.html`

**Obsoletos (podem ficar, não atrapalham; ou apagar):** `radar-binance-passo1.html` a `passo8`, `treino.js`, `saidas.js`, `forca.js`, `survivorship.js`, `modelo.js`, `radar-binance.user.js`.

### 10.2 Fluxos (Actions)
Em **Add file, Create new file**, com o nome exato (sem ponto no fim, sem acento), colar o texto da seção 12:
- `.github/workflows/robo.yml` (já existe; editar com o lápis e substituir tudo)
- `.github/workflows/pesquisa.yml` (novo)

Na aba **Ações**, desativar os fluxos antigos: `treino`, `disse como` (é o `saidas` traduzido), `forca` e `sobrevivência` (toque no nome, nos três pontinhos e em Disable workflow).

Na janela de confirmação de qualquer edição, deixar marcada a primeira opção ("Comprometa-se diretamente com principal").

### 10.3 Rodar
Aba Ações, tocar no nome do fluxo, **Run workflow**, botão verde. Ordem: `pesquisa` (espere ficar verde) e depois `robô`. Cerca de 5 minutos depois, abrir o painel.

### 10.4 Como ler o painel
- Aviso vermelho no topo = algum arquivo ou resultado está em versão antiga: reenviar o arquivo indicado e rodar a pesquisa.
- Cada sinal diz por que o robô entrou ou não.
- As abertas "valem" o preço de agora, mas só contam depois que fecham.

### 10.5 Alertas por Telegram (opcional)
1. No Telegram, procurar `@BotFather`, enviar `/newbot` e guardar o token.
2. Mandar uma mensagem ao bot e procurar `@userinfobot` para ver o número do seu chat.
3. No GitHub: Settings, Secrets and variables, Actions, New repository secret: `TG_TOKEN` (token) e `TG_CHAT` (número do chat).
4. Rodar o `robô` uma vez. Chegam avisos de entrada, saída e bom sinal.
Nunca colar o token em conversas. O robô só envia mensagens.

### 10.6 TradingView
Abaixo de cada moeda há "Ver gráfico", "Tela cheia" (com ferramentas de desenho e troca de moeda) e "Abrir no TradingView (minha conta e meus scripts)". O gráfico embutido não aceita login nem scripts; para usar seus scripts, use o atalho, que abre o mesmo par no site com a sua conta.

---

## 11. Mapa dos arquivos

| Arquivo | Para que serve |
|---|---|
| `bot.js` | Robô de hora em hora: confere operações, procura sinais, calcula barras, usa o modelo, grava `data/*.json`, aprende com resultados, avisa no Telegram. |
| `ml.js` | Biblioteca: atributos, modelo, treino, filtro de oscilação mínima, RSI, ADX, versão. |
| `dados.js` | Baixa o histórico público diário da Binance (inclui moedas que saíram). |
| `pesquisa.js` | Treino das variantes do modelo, regras de saída, força e carteira, detector de véspera e conferência, baixando o histórico uma vez. |
| `nuvem.html` | Painel do robô. |
| `cenarios.html` | Cenários de 7 dias (alta, estabilidade e baixa). |
| `index.html` | Menu. |
| `data/*.json` | Criados sozinhos: `trades`, `shadow`, `predictions`, `model`, `forca`, `saidas`, `vespera`, `conferencia`, `pesquisa`, `last-run`. |

---

## 12. Textos dos fluxos

### `.github/workflows/robo.yml`
```
name: robo
on:
  schedule:
    - cron: "7 * * * *"
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: robo
  cancel-in-progress: false
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: node bot.js
        env:
          TG_TOKEN: ${{ secrets.TG_TOKEN }}
          TG_CHAT: ${{ secrets.TG_CHAT }}
      - run: |
          git config user.name "radar-bot"
          git config user.email "radar-bot@users.noreply.github.com"
          git add data
          git diff --cached --quiet || (git commit -m "dados do robo" && git pull --rebase && git push)
```

### `.github/workflows/pesquisa.yml`
```
name: pesquisa
on:
  workflow_dispatch:
  schedule:
    - cron: "30 3 * * 0"
permissions:
  contents: write
concurrency:
  group: pesquisa
jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: node pesquisa.js
      - run: |
          git config user.name "radar-bot"
          git config user.email "radar-bot@users.noreply.github.com"
          git add data
          git diff --cached --quiet || (git commit -m "pesquisa" && git pull --rebase && git push)
```

---

## 13. Critérios de decisão (resumo)

| Decisão | Critério |
|---|---|
| Modelo "validado" | erro menor que o chute básico, 40+ operações, média positiva, z de 2 ou mais contra o acaso |
| Trocar de variante do modelo | validada e melhor que a base com z de 3 ou mais, no mesmo período |
| Usar o filtro de barras na entrada | só quando o modelo não estiver em uso e as barras separarem bons e ruins com z de 3 ou mais |
| Detector de véspera vale | taxa de acerto 2 vezes maior que a normal (limite inferior 1,5), z de 3 ou mais e média positiva |
| Mudar qualquer regra de entrada | 30+ operações-sombra fechadas, média positiva depois dos custos, z de 3 ou mais |
| Pensar em dinheiro real | tudo acima, e começar com o mínimo, sem alavancagem |
| Modelo rebaixado sozinho | depois de 100 previsões ao vivo, erro pior que o chute básico em mais de 0,02 |

---

## 14. Como continuar em outra conversa (e gastar menos)

- **Crie um Projeto no Claude** (app: Projetos, Novo projeto), dê o nome "Radar Binance", suba `PROJETO.md` e este `RELATORIO.md` como conhecimento, e cole as instruções do fim do `PROJETO.md` no campo de instruções. Abra uma conversa nova a cada etapa.
- Primeira mensagem sugerida: "continuar: li o relatório; a próxima pendência é X" (ou só "continuar").
- Nas conversas novas, mande só a gravação ou captura do trecho que importa, e peça respostas curtas.
- Se o limite do plano interromper uma resposta, escreva "continuar" quando voltar. O Claude não retoma sozinho.
- O plano grátis renova o limite por sessões de algumas horas; conversas longas gastam muito mais, porque cada mensagem reprocessa o histórico inteiro.

---

## 15. Glossário curto

- **Candle:** barra de preços de um período (aqui, 1 dia ou 1 hora).
- **ATR / oscilação diária:** o quanto a moeda costuma se mexer por dia, em %.
- **Stop:** saída automática com prejuízo limitado. **Meta:** saída com lucro.
- **OCO / ordem limite:** ordens de venda programadas na Binance para capturar a meta ou o stop sem você estar olhando.
- **Operação-sombra:** registro do que teria acontecido em cada sinal de uma regra, para juntar amostra mais rápido; não é operação real.
- **z:** quantos "desvios" a diferença é maior que o acaso; 2 é mínimo, 3 é mais seguro.
- **Margem de erro (±):** o quanto a média pode variar só por sorte.
- **Fora da amostra:** testar em dados que não foram usados para criar a regra.
- **Viés de sobrevivência:** ignorar moedas que sumiram deixa os resultados mais bonitos do que a realidade.
- **Sinal esticado:** preço que já subiu demais desde o sinal; entrar seria correr atrás dele.
