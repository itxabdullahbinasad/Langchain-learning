#include<iostream>
using namespace std;
class TaxPayer{
protected:
long TaxPayable;
double Annual_income;
float TaxRate;
public:
TaxPayer(){
	TaxPayable=0.0;
	Annual_income=0;
	TaxRate=0.0;
}

virtual void calculate_Tax(){
}
};
class Salaried_Person:public TaxPayer{

	public:
	Salaried_Person(double AI){
		Annual_income=AI;
	}

	void calculate_Tax(){
		if(Annual_income<=600000)
		{
			TaxRate=0;
			TaxPayable = (Annual_income * TaxRate);
		}
		else if ((Annual_income>=600001) && (Annual_income<=1200000))
		{
			TaxRate=0.05;
			TaxPayable = (Annual_income * TaxRate);
		}
		else if ((Annual_income>=1200001) && (Annual_income<=2400000))
		{
			TaxRate=0.1;
			TaxPayable = (Annual_income * TaxRate);
		}
		else if ((Annual_income>=2400001) && (Annual_income<=4800000))
		{
			TaxRate=0.15;
			TaxPayable = (Annual_income * TaxRate);
		}
			
		else if ((Annual_income>=4800001) && (Annual_income<=9600000))
		{
			TaxRate=0.2;
			TaxPayable = (Annual_income * TaxRate);
		}	
		
		else if (Annual_income>9600000)
		{
			TaxRate=0.25;
			TaxPayable = (Annual_income * TaxRate);
		}
		cout<<endl<<"For salaried Person, Tax Payable Amount = "<< TaxPayable;
	}
	
};
class Business_Person:public TaxPayer{
double Allowable_Expenses;
public:
	Business_Person(double AI){
		Annual_income=AI;
		Allowable_Expenses=Annual_income*0.6;
	}
	void calculate_Tax(){
		if(Annual_income>=1000000)
		{
			TaxRate=0.29;
		TaxPayable = (Annual_income - Allowable_Expenses) * TaxRate;
		cout<<endl<<"For Business Person, Tax Payable Amount = "<< TaxPayable;
		cout<<endl;
		}
		
		else
		{
			TaxPayable=0;
			cout<<endl<<"For Business Person, Minimum Annual Income is 1,000,000, so, Tax Payable Amount = "<< TaxPayable;
			cout<<endl; 
		}
	}
		
		
};

int main()
{
	int no_users;
	cout<<"----------------------------------------------"<<endl;
	cout<<"-------------------TAX CALCULATOR-------------"<<endl;
	cout<<"----------------------------------------------"<<endl<<endl;
	no_users=2;
	double Annualincome;
	cout<<"Enter Annual Income: ";
	cin>>Annualincome;

	TaxPayer* users[no_users];
	users[0]=new Business_Person(Annualincome);
	users[1]=new Salaried_Person(Annualincome);
	for (int i = 0; i < 2; ++i) {
				
        		users[i]->calculate_Tax();		
	}

    
return 0;
}


