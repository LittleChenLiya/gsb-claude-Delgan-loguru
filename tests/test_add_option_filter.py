import re

import pytest

from loguru import logger


@pytest.mark.parametrize(
    "filter",
    [
        None,
        "",
        "tests",
        "tests.test_add_option_filter",
        (lambda r: True),
        (lambda r: r["level"].name == "DEBUG"),
        {},
        {"": "DEBUG"},
        {"tests": True},
        {"tests.test_add_option_filter": 10},
        {"": "WARNING", "tests": 0},
        {"tests.test_add_option_filter": 5, "tests": False},
        {"tests.test_add_option_filter.foobar": False},
        {"tests.": False},
        {"tests.test_add_option_filter.": False},
    ],
)
def test_filtered_in(filter, writer):
    logger.add(writer, filter=filter, format="{message}")
    logger.debug("Test Filter")
    assert writer.read() == "Test Filter\n"


@pytest.mark.parametrize(
    "filter",
    [
        "test",
        "testsx",
        "tests.",
        "tests.test_add_option_filter.",
        ".",
        (lambda r: False),
        (lambda r: r["level"].no != 10),
        {"": False},
        {"": True, "tests": 50},
        {"tests.test_add_option_filter": False},
        {"tests": "WARNING"},
        {"tests": 5, "tests.test_add_option_filter": 40},
        {"": 100, "tests.test_add_option_filter.foobar": True},
    ],
)
def test_filtered_out(filter, writer):
    logger.add(writer, filter=filter, format="{message}")
    logger.debug("Test Filter")
    assert writer.read() == ""


@pytest.mark.parametrize(
    "filter",
    [
        None,
        lambda _: True,
        {},
        {None: 0},
        {"": False},
        {"tests": False, None: True},
        {"unrelated": 100},
        {None: "INFO", "": "WARNING"},
    ],
)
def test_filtered_in_incomplete_frame_context(writer, filter, incomplete_frame_context):
    logger.add(writer, filter=filter, format="{message}", catch=False)
    logger.info("It's ok")
    assert writer.read() == "It's ok\n"


@pytest.mark.parametrize(
    "filter",
    [
        "tests",
        "",
        lambda _: False,
        {None: False},
        {"": 0, None: "WARNING"},
        {None: 100, "tests": True},
    ],
)
def test_filtered_out_incomplete_frame_context(writer, filter, incomplete_frame_context):
    logger.add(writer, filter=filter, format="{message}", catch=False)
    logger.info("It's not ok")
    assert writer.read() == ""


@pytest.mark.parametrize("filter", [-1, 3.4, object()])
def test_invalid_filter(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{"foo": None}, {"foo": 2.5}, {"a": "DEBUG", "b": object()}])
def test_invalid_filter_dict_level_types(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{1: "DEBUG"}, {object(): 10}])
def test_invalid_filter_dict_module_types(writer, filter):
    with pytest.raises(TypeError):
        logger.add(writer, filter=filter)


@pytest.mark.parametrize("filter", [{"foo": "UNKNOWN_LEVEL"}, {"tests.test_add_option_filter": ""}])
def test_invalid_filter_dict_values_unknown_level(writer, filter):
    with pytest.raises(
        ValueError,
        match=(
            r"The filter dict contains a module '[^']*' associated to "
            r"a level name which does not exist: '[^']*'"
        ),
    ):
        logger.add(writer, filter=filter)


def test_invalid_filter_dict_values_wrong_integer_value(writer):
    with pytest.raises(
        ValueError,
        match=(
            r"The filter dict contains a module '[^']*' associated to an invalid level, "
            r"it should be a positive integer, not: '[^']*'"
        ),
    ):
        logger.add(writer, filter={"tests": -1})


def test_filter_dict_with_custom_level(writer):
    logger.level("MY_LEVEL", 6, color="", icon="")
    logger.add(writer, level=0, filter={"tests": "MY_LEVEL"}, format="{message}")
    logger.log(3, "No")
    logger.log(9, "Yes")
    assert writer.read() == "Yes\n"


def test_invalid_filter_builtin(writer):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The built-in 'filter()' function cannot be used as a 'filter' parameter, this is "
            "most likely a mistake (please double-check the arguments passed to 'logger.add()'"
        ),
    ):
        logger.add(writer, filter=filter)


def broken_filter(record):
    raise ValueError("Filter error!")


def test_filter_error_catch_filter_true(writer, capsys):
    received = []

    def other_sink(message):
        received.append(message)

    handler_id = logger.add(
        writer, filter=broken_filter, format="{message}", catch=False, catch_filter=True
    )
    logger.add(other_sink, format="{message}")
    logger.info("Test")

    out, err = capsys.readouterr()
    lines = err.strip().splitlines()

    assert writer.read() == ""
    assert received == ["Test\n"]
    assert out == ""
    assert lines[0] == "--- Logging error in Loguru Handler #%d ---" % handler_id
    assert re.match(r"Record was: \{.*Test.*\}", lines[1])
    assert lines[-2].startswith("ValueError: Filter error!")
    assert lines[-1] == "--- End of logging error ---"


def test_filter_error_catch_filter_false(writer, capsys):
    logger.add(writer, filter=broken_filter, format="{message}", catch=True, catch_filter=False)
    with pytest.raises(ValueError, match="Filter error!"):
        logger.info("Test")

    out, err = capsys.readouterr()
    assert writer.read() == ""
    assert out == err == ""


def test_filter_error_default_catch_is_true(writer, capsys):
    logger.add(writer, filter=broken_filter, format="{message}", catch=True)
    logger.info("Test")

    out, err = capsys.readouterr()
    lines = err.strip().splitlines()

    assert writer.read() == ""
    assert out == ""
    assert lines[0] == "--- Logging error in Loguru Handler #0 ---"
    assert lines[-2].startswith("ValueError: Filter error!")
    assert lines[-1] == "--- End of logging error ---"


def test_filter_error_default_catch_is_false(writer, capsys):
    logger.add(writer, filter=broken_filter, format="{message}", catch=False)
    with pytest.raises(ValueError, match="Filter error!"):
        logger.info("Test")

    out, err = capsys.readouterr()
    assert writer.read() == ""
    assert out == err == ""


def test_filter_error_not_logged_through_handlers(writer, capsys):
    received = []
    calls = 0

    def counting_broken_filter(record):
        nonlocal calls
        calls += 1
        raise ValueError("Filter error!")

    def other_sink(message):
        received.append(message)

    logger.add(writer, filter=counting_broken_filter, format="{message}", catch_filter=True)
    logger.add(other_sink, format="{message}")
    logger.info("Test")

    out, err = capsys.readouterr()

    # The filter error is reported on stderr only: it is not wrapped into a new log
    # record, so the other sink sees the original message exactly once and the faulty
    # filter is not re-triggered (no recursion).
    assert received == ["Test\n"]
    assert calls == 1
    assert writer.read() == ""
    assert out == ""
    assert "ValueError: Filter error!" in err


def test_invalid_catch_filter_value(writer):
    with pytest.raises(
        TypeError,
        match=re.escape("Invalid 'catch_filter', it should be a boolean or None, not: 'str'"),
    ):
        logger.add(writer, catch_filter="yes")
