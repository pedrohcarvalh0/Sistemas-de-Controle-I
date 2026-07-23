"""
============================================================================
 SIMULACAO DAS REPRESENTACOES NO ESPACO DE ESTADOS  ·  Prob. 5.80 (HIV/AIDS)
 Cap. 5 - Reducao de Subsistemas Multiplos (Nise, 6a ed.)
============================================================================

OBJETIVO
--------
Mostrar que um MESMO sistema pode ser escrito de varias formas no espaco de
estados (5.7) e que uma transformacao de similaridade / diagonalizacao (5.8)
nao altera o comportamento entrada -> saida. Provamos isso numericamente:
todas as formas produzem a mesma resposta ao degrau.

COMO O CODIGO ESTA ORGANIZADO (arquitetura em camadas + SOLID)
-------------------------------------------------------------
Cada "peca" faz UMA coisa so e nao conhece detalhes das outras. Assim da para
trocar/estender qualquer parte sem quebrar o resto.

  Camada 1  MODELO ............ o dado puro: as matrizes A, B, C, D
  Camada 2  REPRESENTACOES .... "estrategias" que constroem cada forma
  Camada 3  SIMULACAO ......... roda a resposta ao degrau de qualquer modelo
  Camada 4  VERIFICACAO ....... compara duas respostas e mede o erro
  Camada 5  EXPORTACAO ........ salva JSON / grafico / GIF (saidas isoladas)
  Camada 6  ORQUESTRACAO ...... o "maestro" que liga tudo (injeta dependencias)

Mapeamento dos principios SOLID (onde cada um aparece):
  S (Responsabilidade unica) - cada classe tem um unico motivo para mudar.
  O (Aberto/Fechado) ......... novas formas entram como novas classes, sem
                               editar as existentes (basta registrar na lista).
  L (Substituicao de Liskov) . toda Representacao pode substituir outra: a
                               orquestracao trata todas pela mesma interface.
  I (Segregacao de interface). exportadores separados (JSON, GIF, PNG); quem
                               so quer JSON nao depende de matplotlib.
  D (Inversao de dependencia). a orquestracao depende de ABSTRACOES (a interface
                               Representacao, a interface Exportador), nao de
                               classes concretas.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import json

import numpy as np
import control as ct


# ===========================================================================
# CAMADA 1 - MODELO   (entidade de dados: so guarda e descreve o sistema)
# ===========================================================================
@dataclass
class ModeloEspacoDeEstados:
    """
    Representa UM sistema no espaco de estados: x' = A x + B u ,  y = C x + D u.

    Responsabilidade unica (S): apenas armazenar as matrizes e oferecer
    consultas basicas derivadas delas (funcao de transferencia, autovalores).
    Nao sabe simular, nem desenhar, nem salvar arquivo - isso e de outros.
    """
    nome: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray = field(default_factory=lambda: np.array([[0.0]]))

    def como_sistema_control(self) -> ct.StateSpace:
        """Converte para o objeto da biblioteca `control` (usado na simulacao)."""
        return ct.ss(self.A, self.B, self.C, self.D)

    def funcao_transferencia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai G(s) = Y(s)/U(s) e devolve (numerador, denominador)."""
        G = ct.ss2tf(self.como_sistema_control())
        num = np.array(G.num[0][0], dtype=float)
        den = np.array(G.den[0][0], dtype=float)
        return num, den

    def autovalores(self) -> np.ndarray:
        """Autovalores de A = polos do sistema (usado na diagonalizacao 5.8)."""
        return np.linalg.eigvals(self.A)


# ===========================================================================
# CAMADA 2 - REPRESENTACOES   (padrao Strategy: uma classe por "forma")
# ===========================================================================
#
# A ideia central do 5.7/5.8: existem varias maneiras de escrever o mesmo
# sistema. Modelamos cada maneira como uma ESTRATEGIA que sabe se construir a
# partir de um modelo-base. Todas obedecem a mesma interface `Representacao`,
# entao a orquestracao pode tratar qualquer uma sem saber qual e (Liskov).
#
# Para adicionar uma nova forma no futuro (ex.: forma em cascata), basta criar
# uma nova subclasse e coloca-la na lista - nenhum codigo antigo muda (Aberto/
# Fechado).
# ---------------------------------------------------------------------------

class Representacao(ABC):
    """Interface comum a todas as formas (a ABSTRACAO da qual tudo depende)."""

    #: rotulo legivel usado em graficos e relatorios
    nome: str = "abstrata"

    @abstractmethod
    def construir(self, base: ModeloEspacoDeEstados) -> ModeloEspacoDeEstados:
        """Recebe o sistema original e devolve ESTE mesmo sistema nesta forma."""
        raise NotImplementedError


# --- Funcoes auxiliares PURAS (sem estado) compartilhadas pelas estrategias ---
# Sao utilitarios matematicos, nao "regras de negocio"; ficam de fora das
# classes para nao duplicar codigo e para serem testaveis isoladamente.

def _coeficientes_normalizados(num: np.ndarray, den: np.ndarray):
    """
    Deixa o denominador MONICO (s^n + a_{n-1}s^{n-1} + ... + a0) e devolve:
      a = [a0, a1, ..., a_{n-1}]   (coeficientes do denominador, ordem crescente)
      b = [b0, b1, ..., b_{n-1}]   (numerador alinhado a mesma base)
      n = ordem do sistema
    """
    den = np.asarray(den, float)
    den = den / den[0]
    n = len(den) - 1
    a = den[1:][::-1]
    b = np.zeros(n)
    num = np.asarray(num, float)
    b[:len(num)] = num[::-1]
    return a, b, n


class SistemaOriginal(Representacao):
    """Forma trivial: devolve o proprio sistema-base, sem transformar nada.
    Serve de referencia para a verificacao de equivalencia."""
    nome = "Original"

    def construir(self, base):
        return ModeloEspacoDeEstados(self.nome, base.A, base.B, base.C, base.D)


class VariaveisDeFase(Representacao):
    """5.7 - Variaveis de fase: matriz companheira INFERIOR (coeficientes na
    ultima linha de A). E o ponto de partida das formas canonicas."""
    nome = "Variáveis de fase"

    def construir(self, base):
        a, b, n = _coeficientes_normalizados(*base.funcao_transferencia())
        A = np.zeros((n, n))
        A[:-1, 1:] = np.eye(n - 1)      # cadeia de integradores (1's acima da diag.)
        A[-1, :] = -a                   # coeficientes do denominador na ultima linha
        B = np.zeros((n, 1)); B[-1, 0] = 1.0
        C = b.reshape(1, n)             # numerador vira a saida
        return ModeloEspacoDeEstados(self.nome, A, B, C, base.D)


class CanonicaControlavel(Representacao):
    """5.7 - Canonica controlavel: matriz companheira SUPERIOR (coeficientes na
    1a linha). Base do projeto de CONTROLADORES (Cap. 12)."""
    nome = "Canônica controlável"

    def construir(self, base):
        a, b, n = _coeficientes_normalizados(*base.funcao_transferencia())
        A = np.zeros((n, n))
        A[0, :] = -a[::-1]              # coeficientes na 1a linha (ordem invertida)
        A[1:, :-1] = np.eye(n - 1)
        B = np.zeros((n, 1)); B[0, 0] = 1.0
        C = b[::-1].reshape(1, n)
        return ModeloEspacoDeEstados(self.nome, A, B, C, base.D)


class CanonicaObservavel(Representacao):
    """5.7 - Canonica observavel: a DUAL da controlavel. Reaproveitamos a
    controlavel e aplicamos as transposicoes (A^T, C^T, B^T). Base do projeto
    de OBSERVADORES (Cap. 12)."""
    nome = "Canônica observável"

    def construir(self, base):
        # Reuso explicito: a observavel e definida EM TERMOS da controlavel.
        cc = CanonicaControlavel().construir(base)
        return ModeloEspacoDeEstados(self.nome, cc.A.T, cc.C.T, cc.B.T, base.D)


class ModalDiagonal(Representacao):
    """5.8 - Diagonalizacao por autovalores/autovetores.

    P = [autovetores nas colunas]  ->  D = P^-1 A P (diagonal).
    Quando ha polos COMPLEXOS conjugados a diagonal fica complexa; para simular
    em aritmetica REAL montamos a forma MODAL (bloco 2x2) com [Re(v), Im(v)]."""
    nome = "Modal / diagonal"

    def construir(self, base):
        w, V = np.linalg.eig(base.A)
        # Detecta o primeiro par complexo e monta a base real correspondente.
        if np.any(np.abs(w.imag) > 1e-9):
            idx_complexo = int(np.argmax(np.abs(w.imag)))  # posicao do par complexo
            idx_real = int(np.argmin(np.abs(w.imag)))      # posicao do polo real
            vr, vi = V[:, idx_complexo].real, V[:, idx_complexo].imag
            vreal = V[:, idx_real].real
            P = np.column_stack([vr, vi, vreal])
        else:
            P = V.real                                     # todos reais -> P direto
        Pinv = np.linalg.inv(P)
        A = Pinv @ base.A @ P
        B = Pinv @ base.B
        C = base.C @ P
        return ModeloEspacoDeEstados(self.nome, A.real, B.real, C.real, base.D)


# ===========================================================================
# CAMADA 3 - SIMULACAO   (roda a resposta ao degrau de qualquer modelo)
# ===========================================================================
@dataclass
class RespostaTemporal:
    """Resultado de uma simulacao: tempo, saida y(t) e estados internos x(t)."""
    t: np.ndarray
    y: np.ndarray            # shape (ntempo,)
    x: np.ndarray            # shape (nestados, ntempo)


class SimuladorDegrau:
    """
    Responsabilidade unica (S): dado um modelo, calcular a resposta ao degrau.

    Depende apenas da ABSTRACAO `ModeloEspacoDeEstados` (D): nao importa qual
    forma foi usada para gerar o modelo - ele simula qualquer uma.
    """
    def __init__(self, t: np.ndarray):
        self._t = t          # base de tempo compartilhada por todas as simulacoes

    def simular(self, modelo: ModeloEspacoDeEstados) -> RespostaTemporal:
        resp = ct.step_response(modelo.como_sistema_control(),
                                T=self._t, return_x=True)
        y = np.squeeze(np.asarray(resp.outputs))
        x = np.asarray(resp.states)
        if x.ndim == 3:      # (nestados, nentradas, ntempo) -> pega a unica entrada
            x = x[:, 0, :]
        return RespostaTemporal(self._t, y, x)


# ===========================================================================
# CAMADA 4 - VERIFICACAO   (mede o quanto duas respostas coincidem)
# ===========================================================================
class VerificadorDeEquivalencia:
    """SRP: unico proposito e comparar respostas e devolver o erro maximo.
    E o teste numerico que prova o argumento teorico da apresentacao."""

    @staticmethod
    def erro_maximo(referencia: RespostaTemporal, outra: RespostaTemporal) -> float:
        return float(np.max(np.abs(outra.y - referencia.y)))


# ===========================================================================
# CAMADA 5 - EXPORTACAO   (saidas isoladas; interfaces segregadas - ISP)
# ===========================================================================
#
# Cada exportador cuida de UM formato. Quem so precisa do JSON (o HTML) nao
# carrega matplotlib; quem quer o GIF nao mexe em JSON. Todos consomem os
# mesmos dados ja simulados, sem saber como foram gerados.
# ---------------------------------------------------------------------------

@dataclass
class ItemSimulado:
    """Pacote pronto para exportar: o modelo de uma forma + sua resposta."""
    modelo: ModeloEspacoDeEstados
    resposta: RespostaTemporal


class ExportadorJSON:
    """Serializa os resultados para o front-end HTML interativo."""

    def exportar(self, itens: List[ItemSimulado], base: ModeloEspacoDeEstados,
                 caminho: str) -> None:
        t = itens[0].resposta.t
        num, den = base.funcao_transferencia()
        dados = {
            "t": [round(float(v), 4) for v in t],
            "tf": {"num": [round(float(v), 4) for v in num],
                   "den": [round(float(v), 4) for v in den]},
            "eig": [[round(float(v.real), 4), round(float(v.imag), 4)]
                    for v in base.autovalores()],
            "forms": [{
                "name": it.modelo.nome,
                "A": self._mat(it.modelo.A),
                "B": self._vec(it.modelo.B),
                "C": self._vec(it.modelo.C),
                "y": self._vec(it.resposta.y),
                "x": [self._vec(xi) for xi in it.resposta.x],
            } for it in itens],
        }
        with open(caminho, "w") as f:
            json.dump(dados, f)

    @staticmethod
    def _mat(M):
        return [[round(float(v), 4) for v in row] for row in np.asarray(M)]

    @staticmethod
    def _vec(v):
        return [round(float(x), 4) for x in np.asarray(v).reshape(-1)]


class GeradorGraficoEstatico:
    """Salva um PNG com todas as respostas sobrepostas (a 'prova' estatica)."""

    def gerar(self, itens: List[ItemSimulado], caminho: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        estilos = ["-", "--", "-.", ":", (0, (5, 1))]
        larguras = [6, 4.5, 3.2, 2.2, 1.2]
        plt.figure(figsize=(10, 6))
        for it, ls, lw in zip(itens, estilos, larguras):
            plt.plot(it.resposta.t, it.resposta.y, linestyle=ls, lw=lw,
                     label=it.modelo.nome)
        plt.title("Todas as representações = mesmo sistema (resposta ao degrau)")
        plt.xlabel("tempo (s)"); plt.ylabel("y(t) = carga viral (saída)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(caminho, dpi=130)
        plt.close()


class GeradorAnimacaoGIF:
    """Desenha as curvas progressivamente e salva um GIF (versao 'dinamica')."""

    def __init__(self, frames_por_forma: int = 28, fps: int = 18):
        self._fpf = frames_por_forma
        self._fps = fps

    def gerar(self, itens: List[ItemSimulado], caminho: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        t = itens[0].resposta.t
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-2800, 300)
        ax.set_xlabel("tempo (s)"); ax.set_ylabel("y(t) = saída")
        ax.grid(alpha=0.3)
        linhas = [ax.plot([], [], color=c, lw=2.4, label=it.modelo.nome)[0]
                  for c, it in zip(cores, itens)]
        titulo = ax.set_title("")
        ax.legend(loc="lower right", fontsize=9)

        total = self._fpf * len(itens)

        def update(frame):
            idx = frame // self._fpf
            prog = (frame % self._fpf + 1) / self._fpf
            for i in range(idx):                     # formas ja concluidas
                linhas[i].set_data(t, itens[i].resposta.y)
            yy = itens[idx].resposta.y               # forma atual: parcial
            k = max(2, int(prog * len(t)))
            linhas[idx].set_data(t[:k], yy[:k])
            titulo.set_text(f"Desenhando: {itens[idx].modelo.nome}  "
                            f"({idx+1}/{len(itens)})")
            return linhas + [titulo]

        anim = FuncAnimation(fig, update, frames=total, blit=False, interval=60)
        anim.save(caminho, writer=PillowWriter(fps=self._fps))
        plt.close()


# ===========================================================================
# CAMADA 6 - ORQUESTRACAO   (o 'maestro': injecao de dependencias - DIP)
# ===========================================================================
class AplicacaoSimulacao:
    """
    Liga todas as pecas SEM conhecer os detalhes internos delas.

    Recebe por INJECAO (parametros do construtor):
      - o modelo-base,
      - a lista de representacoes (abstracoes) a construir,
      - o simulador,
      - a lista de exportadores.
    Trocar qualquer peca (outra forma, outro simulador, outro exportador) nao
    exige mudar esta classe - so a montagem la no `main`.
    """
    def __init__(self, base: ModeloEspacoDeEstados,
                 representacoes: List[Representacao],
                 simulador: SimuladorDegrau):
        self._base = base
        self._representacoes = representacoes
        self._simulador = simulador
        self._itens: List[ItemSimulado] = []

    def executar(self) -> List[ItemSimulado]:
        """Constroi cada forma e simula sua resposta ao degrau."""
        self._itens = []
        for rep in self._representacoes:
            modelo = rep.construir(self._base)          # polimorfismo (Liskov)
            resposta = self._simulador.simular(modelo)
            self._itens.append(ItemSimulado(modelo, resposta))
        return self._itens

    def relatar_equivalencia(self) -> None:
        """Imprime o erro de cada forma em relacao a referencia (a 1a da lista)."""
        ref = self._itens[0].resposta
        print("\nCHECAGEM DE EQUIVALENCIA (erro maximo vs. referencia):")
        for it in self._itens:
            erro = VerificadorDeEquivalencia.erro_maximo(ref, it.resposta)
            print(f"  {it.modelo.nome:24s}: {erro:.3e}")

    def exportar_com(self, exportadores: list, caminhos: dict) -> None:
        """Aplica cada exportador (ISP): passa a lista simulada e o caminho."""
        for exp in exportadores:
            nome = type(exp).__name__
            if isinstance(exp, ExportadorJSON):
                exp.exportar(self._itens, self._base, caminhos["json"])
            elif isinstance(exp, GeradorGraficoEstatico):
                exp.gerar(self._itens, caminhos["png"])
            elif isinstance(exp, GeradorAnimacaoGIF):
                exp.gerar(self._itens, caminhos["gif"])
            print(f"  [ok] {nome} -> gerado")


# ===========================================================================
# MONTAGEM (composicao) - o unico lugar que conhece as classes concretas
# ===========================================================================
def construir_modelo_hiv() -> ModeloEspacoDeEstados:
    """Sistema-base: modelo linearizado do HIV (Craig, 2004) - Prob. 5.80."""
    A = np.array([[-0.04167, 0.0, -0.0058],
                  [0.0217, -0.24, 0.0058],
                  [0.0, 100.0, -2.4]])
    B = np.array([[5.2], [-5.2], [0.0]])
    C = np.array([[0.0, 0.0, 1.0]])
    return ModeloEspacoDeEstados("Original", A, B, C)


def main():
    # 1) modelo-base
    base = construir_modelo_hiv()

    # 2) quais formas queremos (Aberto/Fechado: e so acrescentar/remover aqui)
    representacoes = [
        SistemaOriginal(),
        VariaveisDeFase(),
        CanonicaControlavel(),
        CanonicaObservavel(),
        ModalDiagonal(),
    ]

    # 3) simulador com a base de tempo desejada
    simulador = SimuladorDegrau(t=np.linspace(0, 120, 200))

    # 4) monta a aplicacao injetando as dependencias (DIP)
    app = AplicacaoSimulacao(base, representacoes, simulador)
    app.executar()
    app.relatar_equivalencia()

    # 5) exporta nos formatos desejados (ISP: escolha livre de exportadores)
    print("\nEXPORTANDO:")
    app.exportar_com(
        exportadores=[ExportadorJSON(), GeradorGraficoEstatico(),
                      GeradorAnimacaoGIF()],
        caminhos={"json": "p80_data.json",
                  "png": "p80_step.png",
                  "gif": "p80_anim.gif"},
    )
    print("\nConcluido.")


if __name__ == "__main__":
    main()