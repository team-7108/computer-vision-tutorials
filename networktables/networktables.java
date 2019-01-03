package org.usfirst.frc7108.Robot;
import edu.wpi.first.wpilibj.networktables.NetworkTable;

public class Robot extends TimedRobot 
{
	public static NetworkTable table;

    @Override
    public void robotInit() 
    {
    	table = NetworkTable.getTable("datatable");
    }
    @Override
	public void teleopPeriodic() 
	{
		int x = 1;
		// For co-processor to read
		table.putNumber("X",x);
		// 0.0 is default, when no data is available
		// Read data from co-processor
		int y = table.getNumber("Y", 0.0); 
	}

