from datetime import datetime, time
import pandas as pd
import numpy as np

import dash
from dash import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


design_colors = {'blue': '#0073e0',
                 'green': '#038761',
                 'yellow': '#ffd45c',
                 'orange': '#feb221',
                 'red': '#f85359',
                 'darkBlue': '#01193f',
                 'grey': '#575766'}

# chart colors
column_colors = {'Load (kW)': 'darkBlue',
                 'Solar Generation (kW)': 'orange',
                 'Storage Power (kW)': 'green',
                 'State of Charge (kWh)': 'red',
                 'Net Load (kW)': 'blue'}

# read in data set parsing dates and assigning index
df = pd.read_csv('./ESDS data set.csv',
                 parse_dates={'datetime': ['Year','Month','Day','Hour','Minute']},  # parse date/time fields to datetime column
                 index_col='datetime',                                              # set index to the datetime column
                 date_parser=lambda x: datetime.strptime(x, '%Y %m %d %H %M')       # convert the parsed datetime into datetime
                 )

df.drop(['Unnamed: 0'], axis=1, inplace=True) # drop unneeded row num column

class EnergyStorage:
    """
    Energy Storage (e.g. batteries) are bi-directional, they can be both a sink and source of power.
    `self.update_state_of_charge` is the primary method used to change the state of the EnergyStorage object
    once instantiated
    """
    # state attributes
    eff_losses = 0.
    cycle_count = 0.

    def __init__(self, max_power: float, max_capacity: float, min_capacity: float, effeciency: float):
        self.max_power = max_power # kW, positive value
        self.effeciency = np.sqrt(effeciency)  # 0-1, convert rount-trip efficiency to one-way (assume symetrical)

        self.max_capacity = max_capacity  # kWh
        self.min_capacity = min_capacity  # kWh

        # initialize with a full state of charge
        self.state_of_charge = self.max_capacity  # kWh, defaults to max_capacity

    @property
    def head_room(self) -> float:
        """
        kWh to full state of charge
        """
        return self.max_capacity - self.state_of_charge

    @property
    def usable_energy(self) -> float:
        """
        kWh of available energy based on current state-of-charge
        """
        return self.state_of_charge - self.min_capacity

    def _update_cycle_count(self, energy_delta):
        """
        only count discharged energy towards cycle count
        called by self.update_state_of_charge
        """
        if energy_delta < 0:
            self.cycle_count = self.cycle_count + 1

    def update_state_of_charge(self, requested_power: float, time_step: int) -> float:
        """
        return actual power delivered (kW) based on the `requested_power` (kW) and `time_step` (seconds).
        By convention, discharge is positive analogous to generation (e.g. PV), and charge power is negative.
        """
        # calculate energy delta converting kW to kWh
        # energy delta is negative when requested power is positive (discharging decreases available energy)
        energy_delta = -requested_power * (time_step/3600)

        if requested_power < 0 and self.state_of_charge < self.max_capacity:
            # charge the battery

            # check to ensure the energy delta will not surpass the battery's max cap
            # if it does, then set delta to the head_room left in battery and calculate the power using that
            if self.head_room < np.abs(energy_delta):
                energy_delta = self.head_room
                actual_power = -energy_delta / (time_step/3600)
            else:
                actual_power = requested_power

        # check to ensure the energy delta will not bring the battery below its min cap
        # if it does, then set delta to the usable_energy left in battery and calculate the power using that
        elif requested_power > 0 and self.state_of_charge > self.min_capacity:
            # discharging

            if self.usable_energy < np.abs(energy_delta):
                energy_delta = -self.usable_energy
                actual_power = -energy_delta / (time_step/3600)
            else:
                actual_power = requested_power

        else:
            # skip updating state and return zero actual power
            return 0.0

        # update state of charge
        self.state_of_charge += energy_delta

        # increment cycle count
        self._update_cycle_count(energy_delta)

        # return the actual power based on constraints
        return actual_power


app = dash.Dash(__name__)

# frontend layout of app
app.layout = html.Div([
    html.Div([
        html.H2(children="Solar-Battery Storage Simulator"),
        html.Div([
            html.P('Input Battery Max Capacity (kWh)'),
            dcc.Input(id="max-capacity-input", type="number", value=12)]),
        html.Div([
            html.P('Input Battery Min Capacity (kWh)'),
            dcc.Input(id="min-capacity-input", type="number", value=2)])
    ]),
    html.Br(),
    html.Button('Run Model', id='run-model'),
    html.Br(),
    html.Div(id='savings-summary'),
    dcc.Graph(id="all-data-chart"),
    dcc.Graph(id="monthly-totals-chart"),
    dcc.Graph(id="monthly-ranges-chart"),
    dcc.Graph(id="weekly-load-profile"),
    dash_table.DataTable(id='sim-df', columns=[
        {"name": i, "id": i} for i in ['date-time','Load (kW)','Solar Generation (kW)','Storage Power (kW)',
                                       'State of Charge (kWh)','Net Load (kW)']],
                         page_action="native", page_current= 0, page_size= 96,
                         )
])

# # tester
# @app.callback(
#     dash.dependencies.Output('container-button-basic', 'children'),
#     [dash.dependencies.Input('run-model', 'n_clicks')],
#     [dash.dependencies.State('max-capacity-input', 'value')])
# def update_output(n_clicks, value):
#     return 'The input value was "{}" and the button has been clicked {} times'.format(value, n_clicks)

# run model and populate sim_df in output table callback
@app.callback(
    Output('sim-df', 'data'),
    [Input('run-model', 'n_clicks')],
    [State('max-capacity-input', 'value'),
     State('min-capacity-input', 'value')])
def run_ess_sim(n_clicks, maxCapInput, minCapInput):
    if n_clicks == 0: return [{}]
    print('Running model')

    ess = EnergyStorage(max_power=5, max_capacity=maxCapInput, min_capacity=minCapInput, effeciency=0.9)

    # run the EnergyStorage model on the data set
    results = []
    for index, row in df.iterrows():

        load = row['Load (kW)']

        if row['Solar Generation (kWh)'] > 0:
            # if solar gen exists for the interval, calc solar power (kW) by dividing by time interval

            solarGen_kW = row['Solar Generation (kWh)']/0.25
        else:
            solarGen_kW = 0

        # requested power is the home's load minus the available solar power
        # when negative, this means the battery will be charged off the excess solar power
        # when positive, this will discharge the battery to supply the load
        reqPower = load - solarGen_kW

        # send load to battery to either charge or discharge, depending on sign of reqPower
        battery_power = ess.update_state_of_charge(requested_power=reqPower, time_step=900)

        results.append({
            'date-time': index,                                                 # time interval we're on
            'Load (kW)': row['Load (kW)'],                                      # home's demand on the grid
            'Solar Generation (kW)': solarGen_kW,                               # power generated from solar system
            'Storage Power (kW)': battery_power if battery_power > 0 else 0,    # power discharged from the battery
            'State of Charge (kWh)': ess.state_of_charge,                       # current energy capacity of the battery
            'Net Load (kW)': load - (solarGen_kW + battery_power)               # Net load from the grid is the home demand minus their battery supply and solar generation
        })

    sim_df = pd.DataFrame(results)

    return sim_df.to_dict('records')

@app.callback(
    Output('savings-summary', 'children'),
    [Input('sim-df', 'data')])
def calc_energy_savings(data):
    if data == [{}]: return

    sim_df_energy = pd.DataFrame.from_dict(data).copy().set_index('date-time')
    # convert all kW to kWh
    sim_df_energy = sim_df_energy[[col for col in sim_df_energy.columns if 'kWh' not in col]].multiply(0.25)

    sim_df_energy.columns = sim_df_energy.columns.str.replace(" (kW)", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Generation", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Power", "", regex=False)

    # set net to 0 when negative demand (because no negative energy)
    sim_df_energy.loc[sim_df_energy['Net Load']<0,'Net Load'] = 0
    total_load = sim_df_energy['Load'].sum()
    net_load = sim_df_energy[sim_df_energy['Net Load']>0]['Net Load'].sum()
    energy_savings = total_load - net_load

    return [html.H3('Total Energy Demanded:  '+ str(total_load.round(2))+' kWh'),
            html.H3('Total Actual Energy Usage:  '+ str(net_load.round(2))+' kWh'),
            html.H3('Energy Savings:  '+ str(energy_savings.round(2))+' kWh')]

# all data chart
@app.callback(
    Output('all-data-chart', 'figure'),
    [Input('sim-df', 'data')])
def update_all_data_chart(data):
    if data == [{}]: return

    sim_df = pd.DataFrame.from_dict(data).copy().set_index('date-time')
    # show simple daily simulated load with zoomable, scrollable window
    allTraces = []
    for col in sim_df.columns:
        allTraces.append(go.Scatter(x=sim_df.index, y=sim_df[col], name=col, line=dict(color=design_colors[column_colors[col]])))

    layout = dict(title='All Simulated Data',
                  xaxis=dict(rangeselector=dict(x=.8, y=1, buttons=list([
                      dict(count=1, label='1day', step='day'),
                      dict(count=7, label='1w', step='day'),
                      dict(count=1, label='1mo', step='month'),
                      dict(step='all')])),
                             rangeslider=dict(visible=True, thickness=.1),
                             type='date'))
    allFig = go.Figure(dict(data=allTraces, layout=layout))

    return allFig


@app.callback(
    Output('monthly-totals-chart', 'figure'),
    [Input('sim-df', 'data')])
def update_bar_chart(data):
    if data == [{}]: return

    sim_df_energy = pd.DataFrame.from_dict(data).copy().set_index('date-time')
    # convert all kW to kWh
    sim_df_energy = sim_df_energy[[col for col in sim_df_energy.columns if 'kWh' not in col]].multiply(0.25)

    sim_df_energy.columns = sim_df_energy.columns.str.replace(" (kW)", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Generation", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Power", "", regex=False)

    # set net to 0 when negative demand (because no negative energy)
    sim_df_energy.loc[sim_df_energy['Net Load']<0,'Net Load'] = 0

    # resample and sum for total energy use by month
    sim_df_energy.index = pd.to_datetime(sim_df_energy.index)
    sim_df_energy_monthly = sim_df_energy.resample('M').sum()

    # set up traces for bar chart of monthly totals and boxplot of monthly ranges
    column_colors = {'Load': 'darkBlue', 'Solar': 'orange', 'Storage': 'green', 'Net Load': 'blue'}
    barTraces = []
    for col in sim_df_energy_monthly.columns:
        barTraces.append(go.Bar(x=sim_df_energy_monthly.index.month_name(), y=sim_df_energy_monthly[col], name=col, marker_color=design_colors[column_colors[col]]))

    layout = dict(title='Monthly Total Energy Use')
    barFig = go.Figure(dict(data=barTraces, layout=layout))
    barFig.update_yaxes(title=dict(text='Energy (kWh)'))

    return barFig


@app.callback(
    Output('monthly-ranges-chart', 'figure'),
    [Input('sim-df', 'data')])
def update_box_chart(data):
    if data == [{}]: return

    sim_df_energy = pd.DataFrame.from_dict(data).copy().set_index('date-time')
    # convert all kW to kWh
    sim_df_energy = sim_df_energy[[col for col in sim_df_energy.columns if 'kWh' not in col]].multiply(0.25)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" (kW)", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Generation", "", regex=False)
    sim_df_energy.columns = sim_df_energy.columns.str.replace(" Power", "", regex=False)

    # set net to 0 when negative demand (because no negative energy)
    sim_df_energy.loc[sim_df_energy['Net Load']<0,'Net Load'] = 0

    # resample and sum for total energy use by month
    sim_df_energy.index = pd.to_datetime(sim_df_energy.index)
    sim_df_energy['month_name'] = sim_df_energy.index.month_name()
    sim_df_energy_monthly = sim_df_energy.resample('M').sum()

    # set up traces for bar chart of monthly totals and boxplot of monthly ranges
    column_colors = {'Load': 'darkBlue', 'Solar': 'orange', 'Storage': 'green', 'Net Load': 'blue'}
    boxTraces = []
    for col in sim_df_energy_monthly.columns:
        boxTraces.append(go.Box(x=sim_df_energy.month_name, y=sim_df_energy[col], name=col,
                                marker_color=design_colors[column_colors[col]], boxpoints=False))

    layout = dict(title='Monthly Energy Use Ranges', boxmode='group')
    boxFig = go.Figure(dict(data=boxTraces, layout=layout))
    boxFig.update_yaxes(title=dict(text='Energy (kWh)'))

    return boxFig

@app.callback(
    Output('weekly-load-profile', 'figure'),
    [Input('sim-df', 'data')])
def weekly_load_profile(data):
    if data == [{}]: return

    weekly_lp_data = pd.DataFrame.from_dict(data).copy().set_index('date-time')
    weekly_lp_data.index = pd.to_datetime(weekly_lp_data.index)

    weekly_lp_data['day'] = weekly_lp_data.index.day_name()
    weekly_lp_data['time'] = weekly_lp_data.index.time

    # aggregate data to the day name of week and time period index
    weekly_lp_data_agg = weekly_lp_data.groupby(['day','time']).agg('mean').sort_index()
    weekly_lp_data_agg.reset_index('day', inplace=True)

    # convert the index back to a datetime for ease of graphing
    # use last week's date values as placeholders that we will show as only day name in the graph
    seven_days = pd.date_range('2021-9-5', periods=7)
    load_profile_data = pd.DataFrame()
    for d in seven_days:
        y = weekly_lp_data_agg[weekly_lp_data_agg['day']==d.day_name()].reset_index()
        y.insert(0, 'fake_date', d.date())
        load_profile_data = load_profile_data.append(y)

    # re-index DF on last weeks place holder dates
    load_profile_data['date-time'] = pd.to_datetime(load_profile_data['fake_date'].astype(str)+' '+load_profile_data['time'].astype(str))
    load_profile_data.set_index('date-time',inplace=True)
    load_profile_data.drop(['fake_date','time','day'], axis=1, inplace=True)

    # chart colors
    column_colors = {'Load (kW)': 'darkBlue',
                     'Solar Generation (kW)': 'orange',
                     'Storage Power (kW)': 'green',
                     'State of Charge (kWh)': 'red',
                     'Net Load (kW)': 'blue'}
    lpTraces = []
    for col in load_profile_data.columns:
        lpTraces.append(go.Scatter(x=load_profile_data.index, y=load_profile_data[col], name=col, line=dict(color=design_colors[column_colors[col]])))

    layout = dict(title='Average Weekly Load Profile')
    lpFig = go.Figure(dict(data=lpTraces, layout=layout))
    lpFig.update_yaxes(title=dict(text='Power (kW) [except for battery state in kWh]'))
    lpFig.update_xaxes(tickformat="%A")

    return lpFig


app.run_server(debug=True)