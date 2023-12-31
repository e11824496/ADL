import unittest
from dataset_generation import BankStatementProcessor


DATAFOLDER = 'tests/test_data/'


class TestBankStatementProcessor(unittest.TestCase):
    def test_init(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_no_paypal.csv', DATAFOLDER + 'paypal.csv')

        self.assertEqual(
            processor.bank_statements.shape, (1, 6))

        # Assert that the paypal_data DataFrame is created as expected
        expected_columns = ['Date', 'Time', 'Name', 'Gross',
                            "Transaction ID", "Item Title",
                            "Reference Txn ID", "Bank Reference ID"]

        self.assertTrue(set(processor.paypal_data.columns).issuperset(
            set(expected_columns)))
        self.assertEqual(
            processor.paypal_data.shape[0], 2, msg='Wrong number of rows')

    def test_preprocessBankStatements(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_no_paypal.csv', DATAFOLDER + 'paypal.csv')

        # Call the pre_process_bank_statements method
        processor._preprocess_bank_statements()

        # compare set -> order doesn't matter
        expected_columns = ['Date', 'Time', 'Description', 'Amount']
        self.assertEqual(
            set(processor.bank_statements.columns), set(expected_columns))
        self.assertEqual(
            processor.bank_statements.shape, (1, 4))

        expected_columns = ['Date', 'Time', 'Name', 'Gross',
                            "Transaction ID", "Item Title",
                            "Reference Txn ID", "Bank Reference ID"]

        self.assertTrue(set(processor.paypal_data.columns).issuperset(
            set(expected_columns)))

    def test_is_paypal_payment(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_paypal.csv', DATAFOLDER + 'paypal.csv')
        # Test the is_paypal_payment method with a PayPal payment description
        row = processor.bank_statements.iloc[0]
        self.assertTrue(processor._is_paypal_payment(row))

        # Test with a non-PayPal payment description
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_no_paypal.csv', DATAFOLDER + 'paypal.csv')
        # Test the is_paypal_payment method with a PayPal payment description
        row = processor.bank_statements.iloc[0]
        self.assertFalse(processor._is_paypal_payment(row))

    def test_get_bank_reference(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_paypal.csv', DATAFOLDER + 'paypal.csv')
        text = processor.bank_statements.iloc[0]['Description']
        self.assertEqual(processor._get_bank_reference(text), '0123')

        # Test with no valid bank reference
        text = 'Some other text'
        self.assertIsNone(processor._get_bank_reference(text))

    def test_replace_paypal_payment(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_paypal.csv', DATAFOLDER + 'paypal.csv')
        processor._preprocess_bank_statements()

        row = processor.bank_statements.iloc[0]
        print(processor.paypal_data)
        row = processor._replace_paypal_by_bank_reference(row)
        self.assertIn('Uber Payments BV', row['Description'])

        # Test with a non-PayPal payment description, expect no change
        new_description = row['Description'].replace('0123', '1234')
        row['Description'] = new_description
        row = processor._replace_paypal_by_bank_reference(row)
        self.assertIn(row['Description'], new_description)

    def test_shorten_description(self):
        processor = BankStatementProcessor(
            DATAFOLDER + 'bank_no_paypal.csv', DATAFOLDER + 'paypal.csv')
        processor._preprocess_bank_statements()
        row = processor.bank_statements.iloc[0]

        row = processor._shorten_description(row)
        self.assertEqual(row['Description'],
                         'Verwendungszweck: EUROSPAR DANKT 4141 WIEN 1050')

        # Test with a description that doesn't need shortening
        new_description = 'Zahlungsempfänger: ABC ' + row['Description']
        row['Description'] = new_description
        processor._shorten_description(row)
        self.assertEqual(row['Description'], new_description)

    def tearDown(self):
        pass  # Clean up any resources if needed


if __name__ == '__main__':
    unittest.main()
