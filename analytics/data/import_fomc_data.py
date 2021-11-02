import sys
import datetime
from fomc_data.FomcStatement import FomcStatement
from fomc_data.FomcMinutes import FomcMinutes

batch_id = datetime.date.today().strftime("%y%m%d")

def download_data(fomc, from_year):
    df = fomc.get_contents(from_year)
    print("Shape of the downloaded data: ", df.shape)
    print("The first 5 rows of the data: \n", df.head())
    print("The last 5 rows of the data: \n", df.tail())
    fomc.pickle_dump_df(filename = f"{batch_id}_{fomc.content_type}" + ".pickle")

if __name__ == '__main__':
    pg_name = sys.argv[0]
    args = sys.argv[1:]
    content_type_all = ('statement', 'minutes', 'all')

    if (len(args) != 1) and (len(args) != 2):
        print("Usage: ", pg_name)
        print("Please specify the first argument from ", content_type_all)
        print("You can add from_year (yyyy) as the second argument.")
        print("\n You specified: ", ','.join(args))
        sys.exit(1)

    if len(args) == 1:
        from_year = 1990
    else:
        from_year = int(args[1])
    
    content_type = args[0].lower()
    if content_type not in content_type_all:
        print("Usage: ", pg_name)
        print("Please specify the first argument from ", content_type_all)
        sys.exit(1)
    
    if (from_year < 1980) or (from_year > 2021):
        print("Usage: ", pg_name)
        print("Please specify the second argument between 1980 and 2020")
        sys.exit(1)

    if content_type == 'all':
        fomc = FomcStatement()
        download_data(fomc, from_year)
        fomc = FomcMinutes()
        download_data(fomc, from_year)

    else:
        if content_type == 'statement':
            fomc = FomcStatement()
        elif content_type == 'minutes':
            fomc = FomcMinutes()

        download_data(fomc, from_year)
