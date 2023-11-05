import pandas as pd
import re


class BankStatementProcessor:
    def __init__(self, bank_statement: str, paypal: str) -> None:
        self.bank_columns = ['Date', 'Description',
                             'Date2', 'Amount', 'Currency', 'Datetime']
        self.bank_statements = pd.read_csv(
            bank_statement, sep=';', names=self.bank_columns)

        self.paypal_data = pd.read_csv(paypal, dtype=str)

    def _preprocess_bank_statements(self) -> None:
        # Select relevant columns from bank statements
        relevant_bank_columns = ['Datetime', 'Description', 'Amount']
        self.bank_statements = self.bank_statements[relevant_bank_columns]

        # Extract time from 'Datetime' and drop the original column
        self.bank_statements['Time'] = pd.to_datetime(
            self.bank_statements['Datetime'],
            format='%d.%m.%Y %H:%M:%S:%f').dt.strftime('%H:%M:%S')
        self.bank_statements.drop('Datetime', axis=1, inplace=True)

        # Filter PayPal data to separate bank references
        paypal_references = self.paypal_data[[
            'Zugehöriger Transaktionscode', 'Bankreferenz']
        ].loc[self.paypal_data['Bankreferenz'].notnull()]
        temp = self.paypal_data[['Uhrzeit', 'Name',
                                 'Brutto', 'Transaktionscode',
                                 'Artikelbezeichnung']
                                ]
        self.paypal_data = temp.loc[self.paypal_data['Bankreferenz'].isnull()]                          # noqa: E501

        # Merge multiple PayPal rows based on 'Transaktionscode'
        self.paypal_data = self.paypal_data.merge(
            paypal_references, left_on='Transaktionscode', right_on='Zugehöriger Transaktionscode', how='inner')     # noqa: E501
        self.paypal_data.set_index('Bankreferenz', inplace=True)

    def _is_paypal_payment(self, row: pd.Series) -> bool:
        return 'PayPal' in row['Description']

    def _get_bank_reference(self, text: str) -> str:
        match = re.search(r'Verwendungszweck:\s*(\d+)', text)
        return match.group(1) if match else None

    def _replace_paypal_payment(self, row: pd.Series) -> pd.Series:
        if not self._is_paypal_payment(row):
            return row
        bank_reference = self._get_bank_reference(row['Description'])
        if bank_reference in self.paypal_data.index:
            row = row.copy()

            paypal_row = self.paypal_data.loc[bank_reference]
            row['Description'] = f"Verwendungsweck: PayPal {paypal_row['Name']}"                    # noqa: E501
            if not pd.isnull(paypal_row['Artikelbezeichnung']):
                row['Description'] += f" - {paypal_row['Artikelbezeichnung']}"

            time = pd.to_datetime(paypal_row['Uhrzeit'], format='%H:%M:%S')
            time += pd.Timedelta(hours=8)
            row['Time'] = time.strftime('%H:%M:%S')
        return row

    def _shorten_description(self, row: pd.Series) -> pd.Series:
        if row['Description'].startswith('Verwendungszweck:'):
            row = row.copy()

            row['Description'] = row['Description'].split(
                'Zahlungsreferenz:')[0].strip()
        return row

    def create_statement_string(self, row: pd.Series) -> str:
        return f"{row['Time']}; {row['Description']}; Amount:{row['Amount']}"

    def create_dataset(self) -> None:
        self._preprocess_bank_statements()
        self.bank_statements = self.bank_statements.apply(
            self._replace_paypal_payment, axis=1)
        self.bank_statements = self.bank_statements.apply(
            self._shorten_description, axis=1)
        self.bank_statements_text = self.bank_statements.apply(
            self.create_statement_string, axis=1)

    def save_to_csv(self, output_file: str) -> None:
        self.bank_statements_text.to_csv(output_file, header=True, index=False)


if __name__ == '__main__':
    processor = BankStatementProcessor(
        'BankStatements/export.csv', 'BankStatements/paypal.csv')
    processor.create_dataset()
    processor.save_to_csv('output.csv')
