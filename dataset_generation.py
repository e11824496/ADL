import pandas as pd
import re


class BankStatementProcessor:
    def __init__(self, bank_statement: str, paypal: str) -> None:
        self.bank_columns = ['Date', 'Description',
                             'Date2', 'Amount', 'Currency', 'Datetime']
        self.bank_statements = pd.read_csv(
            bank_statement, sep=';', names=self.bank_columns)

        self.paypal_data = pd.read_csv(paypal, dtype=str, index_col=False)

    def _preprocess_bank_statements(self) -> None:
        """
        Preprocesses the bank statements data by selecting relevant columns,
        converting data types, extracting time and date information,
        and combining multiple PayPal rows from one transaction.
        """
        # Select relevant columns from bank statements
        relevant_bank_columns = ['Datetime', 'Description', 'Amount']
        self.bank_statements = self.bank_statements[relevant_bank_columns]
        self.bank_statements['Amount'] = self.bank_statements['Amount']\
            .str.replace(',', '.')
        self.bank_statements['Amount'] = self.bank_statements['Amount']\
            .astype(float)

        # Extract time from 'Datetime' and drop the original column
        self.bank_statements['Datetime'] = pd.to_datetime(
            self.bank_statements['Datetime'], format='%d.%m.%Y %H:%M:%S:%f')
        self.bank_statements['Time'] = self.bank_statements['Datetime'].dt.time
        self.bank_statements['Date'] = self.bank_statements['Datetime'].dt.date
        self.bank_statements.drop('Datetime', axis=1, inplace=True)

        self.paypal_data['Gross'] = self.paypal_data['Gross'].apply(
            lambda x: float(x.replace('.', '').replace(',', '.')))
        self.paypal_data['Date'] = pd.to_datetime(
            self.paypal_data['Date'], format='%d/%m/%Y')

        paypal_completed = self.paypal_data\
            .loc[self.paypal_data['Status'] == 'Completed']
        paypal_completed = paypal_completed[[
            'Date', 'Time', 'Name', 'Transaction ID', 'Item Title']]

        paypal_possible_bank_references = self.paypal_data[[
            'Reference Txn ID', 'Bank Reference ID', 'Gross']]

        # Merge multiple PayPal rows based on 'Transaction ID'
        self.paypal_data = paypal_completed.merge(
            paypal_possible_bank_references,
            left_on='Transaction ID',
            right_on='Reference Txn ID',
            how='inner')
        self.paypal_data['Gross'] = -self.paypal_data['Gross']
        self.paypal_data.reset_index(inplace=True)

    def _is_paypal_payment(self, row: pd.Series) -> bool:
        return 'PayPal' in row['Description']

    def _get_bank_reference(self, text: str) -> str:
        match = re.search(r'Verwendungszweck:\s*(\d+)', text)
        return match.group(1) if match else None

    def _paypal_to_bankstatement(self, row: pd.Series, paypal_row: pd.Series) -> pd.Series:     # noqa: E501
        row['Description'] = f"Verwendungsweck: PayPal {paypal_row['Name']}"
        if not pd.isnull(paypal_row['Item Title']):
            row['Description'] += f" - {paypal_row['Item Title']}"

        time = pd.to_datetime(paypal_row['Time'], format='%H:%M:%S')
        time += pd.Timedelta(hours=8)
        row['Time'] = time.strftime('%H:%M:%S')

        return row

    def _replace_paypal_by_bank_reference(self, row: pd.Series) -> pd.Series:
        if not self._is_paypal_payment(row):
            return row
        bank_reference = self._get_bank_reference(row['Description'])
        if bank_reference in self.paypal_data['Bank Reference ID'].values:
            row = row.copy()

            match = self.paypal_data['Bank Reference ID'] == bank_reference
            paypal_row = self.paypal_data\
                .loc[match].iloc[0]

            row = self._paypal_to_bankstatement(row, paypal_row)
            # drop the row from the PayPal data
            self.paypal_data.drop(paypal_row.name, inplace=True)
        return row

    def _replace_paypal_by_amount_date(self, row: pd.Series) -> pd.Series:
        if not self._is_paypal_payment(row):
            return row
        bank_date = row['Date']
        bank_amount = row['Amount']

        paypal_rows = self.paypal_data.loc[self.paypal_data['Gross']
                                           == bank_amount]
        if paypal_rows.empty:
            return row
        if paypal_rows.shape[0] == 1:
            paypal_row = paypal_rows.iloc[0]
        else:
            if bank_amount == '-5,00':
                print(paypal_rows)
            # find the PayPal row with the same date
            diff = pd.to_datetime(bank_date) - paypal_rows['Date']
            # get index with minimum dif but positive
            index = diff[diff > pd.Timedelta(0)].idxmin()
            paypal_row = paypal_rows.loc[index]
        row = row.copy()
        row = self._paypal_to_bankstatement(row, paypal_row)
        self.paypal_data.drop(paypal_row.name, inplace=True, axis=0)
        return row

    def _shorten_description(self, row: pd.Series) -> pd.Series:
        if row['Description'].startswith('Verwendungszweck:'):
            row = row.copy()

            row['Description'] = row['Description'].split(
                'Zahlungsreferenz:')[0].strip()
        return row

    def create_dataset(self) -> None:
        """
        Preprocesses bank statements, replaces PayPal references,
        shortens descriptions, and selects relevant columns
        to create a dataset.
        """
        self._preprocess_bank_statements()
        self.bank_statements = self.bank_statements.apply(
            self._replace_paypal_by_bank_reference, axis=1)
        self.bank_statements = self.bank_statements.apply(
            self._replace_paypal_by_amount_date, axis=1)
        self.bank_statements = self.bank_statements.apply(
            self._shorten_description, axis=1)
        self.bank_statements = self.bank_statements[['Time', 'Description', 'Amount']]                   # noqa: E501

    def save_to_csv(self, output_file: str) -> None:
        self.bank_statements.to_csv(output_file, header=True)


if __name__ == '__main__':
    processor = BankStatementProcessor(
        'BankStatements/export.csv', 'BankStatements/paypal.csv')
    processor.create_dataset()
    processor.save_to_csv('BankStatements/output.csv')
