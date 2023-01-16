import random as r

class Agencia():

    def __init__(self, tel, cnpj, numero):
        self.tel = tel
        self._cnpj = cnpj
        self._numero = numero
        self.clientes = []
        self.caixa = 1000000000
        self.emprestimos = []

    def VerificarCaixa(self):
        if self.caixa < 1000000:
            print(f'! Abaixo do nível ideal !\nR$: {self.caixa:,.2f}')
        else:
            print(f'R$: {self.caixa:,.2f}')

    def Emprestimo(self, valor, cpf, juros):
        if self.caixa > valor:
            self.emprestimos.append((valor, cpf, juros))
        else:
            print('! Não permitido !\n ERROR: SALDO INSUFICIENTE.')

    def AdicionarCliente(self, nome, cpf, patrimonio):
        self.clientes.append((nome, cpf, patrimonio))

#criando heranças:
class AgenciaVirtual(Agencia):

    def __init__(self, url):
        self.url = url
        super().__init__(tel, cnpj, '0001')#serve como crtlC e crtlV dos atributos da class original, para não perde-los.
        self.caixa = 1000000


class AgenciaComum(Agencia):

    def __init__(self, tel, cnpj):
        super().__init__(tel, cnpj, numero = r.randint(1000,9999))
        self.caixa = 10000000


class AgenciaVIP(Agencia):

    def __init__(self, tel, cnpj):
        super().__init__(tel, cnpj, numero=r.randint(1000, 9999))
        self.caixa = 1000000

    def AdicionarCliente(self, nome, cpf, patrimonio):
        if patrimonio > 50000000:
            super().AdicionarCliente(nome, cpf, patrimonio)
        else:
            print('Cliente sem requisitos necessários.')


if __name__ == '__main__': #NÃO ESQUECE DE USAR ISSO PARA IMPEDIR O IMPORT IMPORTAR SEU TEST, IDIOTA!

    Agencia01 = Agencia(26351333, 154889785, 1458)
    Agencia01.Emprestimo(12000, 15259783735, 0.02)
    Agencia01.AdicionarCliente('Alexy','15259783735',50000)