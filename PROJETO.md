# Radar Binance: resumo para continuar (atualizado em 07/11/2026)
Leia também o RELATORIO.md (detalhes completos).

## Objetivo e regras combinadas
- Robô de SINAIS e estudo para a Binance Spot, só simulação, capital de R$ 50, lucro líquido modesto. Não é recomendação de compra.
- Dinheiro real só se: 30 ou mais operações-sombra fechadas, média positiva depois dos custos (taxa 0,075%, slippage 0,1%, conversão reais/USDT 0,1% só no começo e no fim) e vantagem segura contra o acaso (z de 3 ou mais). Começar com o mínimo, sem alavancagem, sem colar chaves de API em conversa.
- Formato: "R$ 2,50 (5,0%)". Português do Brasil, prosa simples, sem emojis, no máximo uma pergunta por resposta, respostas curtas.
- Usuário: leigo, só celular (Android, Chrome, Termux). GitHub: warleywelis/radar, site https://warleywelis.github.io/radar/ (nuvem.html = painel). O navegador traduz nomes: "saidas" vira "disse como", "main" vira "principal".

## Como funciona
- bot.js (fluxo robo.yml, de hora em hora): sinal de momentum (fechamento acima da máxima de 20 dias com volume 2x, velas diárias, top 60 por volume médio de 30 dias, oscilação diária mínima de 1,2%), meta 10%, stop de 2 oscilações (4% a 15%), prazo 7 dias, uma operação por vez. Entrada principal = previsão do modelo dos 3 caminhos quando validado (chance de alta de 40% ou mais); senão, momentum. Nunca entra em sinal esticado (mais de 1,5 oscilação acima do sinal). Pausa de 3 dias depois de 3 stops seguidos. Operações-sombra de várias regras e aprendizado contínuo do modelo.
- pesquisa.js (fluxo pesquisa.yml, manual): baixa o histórico público uma vez e roda treino das variantes do modelo, regras de saída, força e carteira de R$ 50, detector de véspera e conferência (robô contra teste histórico). Escreve data/*.json.
- ml.js e dados.js são bibliotecas. nuvem.html lê data/*.json pelo raw.githubusercontent.com. cenarios.html e index.html completam o site. O painel avisa em vermelho quando algum resultado vem de versão antiga.
- Usa data-api.binance.vision (o api.binance.com é bloqueado nos servidores do GitHub).

## Resultados históricos (com filtro de moedas atreladas)
- Modelo validado (07/11): 261 operações, R$ 0,82 (1,6%) ± R$ 0,51 por operação, contra R$ -0,37 (-0,7%) ao acaso, z = 4,54. Calibração ruim acima de 50%. Variantes top 40/60/80, RSI/ADX e persistência passam, nenhuma melhor que a base. Top 100 não passa.
- Saídas: meta fixa de 10% rende R$ 0,25 (0,5%); sem meta R$ 0,72 (1,4%) com 38% de acerto; metas pequenas e stop curto perdem. Regra dinâmica por barras não passou.
- Barras: separam bons de ruins (z = 3,9), sem melhora regular entre 3, 4 e 5. Servem só para filtrar. Carteira de R$ 50: qualquer sinal arruína; 3 barras ou mais ficou no azul com quedas de 29% a 61%.

## Placar real da simulação (até 07/11)
- Principal: 7 fechadas, 3 lucros, total R$ -10,21 (AVA, ALT, BMT e TRX no stop; IOST, INIT e BMT na meta).
- Sombras: momentum com meta 17 fechadas, 41% de acerto, média R$ -2,12 (-4,2%); momentum sem meta 16 fechadas, 1 lucro, média R$ -5,50 (-11,0%); modelo 10 fechadas todas paradas em -0,7% (moedas calmas); barras 3+ janela 5 fechadas, 2 lucros.
- Hipóteses da diferença para o teste: mercado em queda (Bitcoin -7,3% em outubro), moedas escolhidas por volume do dia nas primeiras semanas, modelo prevê "subir 1 oscilação" (favorece moedas calmas) enquanto a meta é 10%, execução do robô.

## Pendências (em ordem)
1. Enviar ao repositório bot.js, ml.js, dados.js, pesquisa.js, nuvem.html, index.html, cenarios.html; criar/atualizar os fluxos robo.yml e pesquisa.yml (textos no RELATORIO.md); desativar os fluxos antigos (treino, "disse como", forca, sobrevivência); rodar pesquisa (40 a 60 minutos) e depois robô.
2. Ler no painel: aviso de versão vazio, seção Conferência, Detector de véspera e a variante "resultado da operação".
3. Coletar 3 a 4 semanas sem mexer nas regras, olhando o painel uma vez por semana. Mudar regra só com 30+ operações-sombra por coorte, média positiva depois dos custos e z de 3 ou mais.
4. Opcional: alertas por Telegram (segredos TG_TOKEN e TG_CHAT).

## Para economizar tokens
Cada mensagem reprocessa a conversa inteira. Trabalhar dentro de um Projeto do Claude com PROJETO.md e RELATORIO.md como conhecimento, abrir conversa nova a cada etapa, dizer só "continuar" para retomar, mandar capturas só do trecho necessário e pedir respostas curtas. O Claude não retoma sozinho depois do limite do plano.

## Instruções do Projeto (colar no campo de instruções)
Você ajuda o Warley a construir e interpretar o Radar Binance (simulação, R$ 50). Leia o PROJETO.md e o RELATORIO.md antes de responder. Responda em português do Brasil, curto e direto, prosa simples, valores no formato "R$ 2,50 (5,0%)", no máximo uma pergunta. Não repita instruções já dadas nem reenvie arquivos sem mudança. Seja honesto: não prometa lucro, diga quando um resultado não prova nada e aplique o critério combinado. Ao retomar ("continuar"), vá direto à próxima pendência.
