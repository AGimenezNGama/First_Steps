from datetime import datetime
import pytz
import random

class ContaCorrente:

    @staticmethod #serve para indicar ao usuário que é apenas um metodo auxiliar, não relacionado ao init e nao usual.
    def DataHora():
        fuso_br = pytz.timezone('Brazil/East')
        horario_br = datetime.now(fuso_br)
        return horario_br.strftime("%d/%m/%Y %H:%M:%S")

    def __init__ (self, nome, cpf, ag, conta):
        self.nome = nome
        self.cpf = cpf
        self.ag = None
        self.conta = None
        self._saldo = 0
        self._lim = None
        self._transacoes = []
        self._cartoes = []

    def _lim(self): #quando há "_" no início da def, significa que não é usual .
        self._lim = -1000
        return self._lim

    def ConsultarSaldo(self): #consulta de saldo atual na conta
        print(f'Seu saldo é de R$: {self._saldo:,.2f}')

    def ConsultarChequeEspecial(self): #consulta do cheque especial atual
        print(f'Seu Cheque especial é de R$: {self._lim():,.2f}')

    def Depositar(self, valor): #deposito na própria conta
        self._saldo += valor
        print(f'Seu saldo atualizado é de R$: {self._saldo:,.2f}')
        self._transacoes.append((valor, self._saldo, ContaCorrente.DataHora()))

    def Sacar(self, valor): #saque e, caso falhe ou não, demonstração de valor.
        if  (self._saldo - valor) < self._lim():
            print(f'Impossível retirar esse valor. (Saldo insuficiente)')
            self.ConsultarSaldo()
        else:
            self._saldo -= valor
            print(f'Seu saldo atualizado é de R$: {self._saldo:,.2f}')
            self._transacoes.append((-valor, self._saldo, ContaCorrente.DataHora()))

    def Extrato(self): #demonstração de um extrato bancário simples com data e hora
        print(f'Extrato bancário: \n{self.nome}\n{self.cpf}\n')
        print('| Saque/Desposito  |' + ' ' *5 + ' Saldo ' + ' ' *5 + ' |    Data & Hora' + ' ' *4 + '|')
        print('-'*59)
        for transacao in self._transacoes:
            print('|{:^18,.2f}|{:^18,.2f}|{:^18}|\n'.format(transacao[0],transacao[1],transacao[2]))

    def Pix(self, valor, conta_destino): #fazer transferência para uma outra conta bancária
        self._saldo -= valor #retira o valor da conta original
        self._transacoes.append((-valor, self._saldo, ContaCorrente.DataHora())) #lança no extrato a retirada da conta original
        conta_destino._saldo += valor #transfere o valor da conta original para a conta_destino
        conta_destino._transacoes.append((valor, conta_destino._saldo, ContaCorrente.DataHora())) #lança no extrato o deposito na conta_destino


class CartaoCredito():


    @staticmethod  # serve para indicar ao usuário que é apenas um metodo auxiliar, não relacionado ao init e nao usual.
    def DataHora():
        fuso_br = pytz.timezone('Brazil/East')
        horario_br = datetime.now(fuso_br)
        return horario_br

    def __init__(self, titular, cc):
        self._numero = random.randint(1000000000000000,9999999999999999)
        self.cc = cc
        self.titular = titular
        self._validade = f'{CartaoCredito.DataHora().month}/{CartaoCredito.DataHora().year + 4}'
        self._cvv = random.randint(100,999)
        self._limite = 0
        self._senha = '0123'
        cc._cartoes.append(self)

    def AlterarLim(self, valor):
        self._limite += valor
        print(f'O limite do seu Cartão: {self._numero} foi atualizado para o valor de R$: {self._limite:,.2f}')

    @property
    def senha(self):
        return self._senha

    @senha.setter
    def senha(self, valor):
        if len(valor) == 4 and valor.isnumeric():
            self._senha = valor
        else:
            print('Nova senha indisponível')