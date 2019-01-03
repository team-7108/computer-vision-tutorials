from networktables import NetworkTables
# To see messages from networktables, you must setup logging
import logging
logging.basicConfig(level=logging.DEBUG)

ip = "10.71.8.2" # ip of roborio
# Init networktables
NetworkTables.initialize(server=ip)
# Get the table
table = NetworkTables.getTable("datatable")
i = 0
while True:
	# Read data from the table
    print('X:', table.getNumber('X', 'N/A'))
    # Write data to table
    table.putNumber('Y', i)
    i += 1
